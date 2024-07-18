import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder, PositionalEncoding1D
from utils.utils import NeighborSampler


class DTFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 node_snap_counts: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2,
                 num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu',
                 feat_using: list = None,
                 num_patch_size=1, intersect_mode='sum'):
        super(DTFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.node_snap_counts = torch.from_numpy(node_snap_counts.astype(np.float32)).to(device)

        self.num_snapshots = self.node_snap_counts.shape[1]

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.snapshot_feat_dim = time_feat_dim
        self.snapshot_encoder = TimeEncoder(time_dim=self.snapshot_feat_dim)

        self.neighbor_intersect_feat_dim = self.channel_embedding_dim

        self.intersect_mode = intersect_mode

        if self.intersect_mode == 'sum':
            self.neighbor_intersect_encoder = NeighborIntersectEncoder(
                neighbor_intersect_feat_dim=self.neighbor_intersect_feat_dim, device=self.device)
        else:
            self.neighbor_intersect_encoder = AppearancesEncoder(
                neighbor_intersect_feat_dim=self.neighbor_intersect_feat_dim,
                num_snapshots=self.num_snapshots, intersect_mode=self.intersect_mode, device=self.device)

        self.num_patch_size = num_patch_size

        self.projection_layer = nn.ModuleDict({
            'pre_snapshot': nn.Sequential(nn.Linear(in_features=1, out_features=self.snapshot_feat_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=self.snapshot_feat_dim,
                                                    out_features=self.snapshot_feat_dim))
        })

        self.transformers_dict = nn.ModuleDict({})

        self.num_channels = len(feat_using) + 2

        for i in range(self.num_patch_size):
            self.projection_layer[f'node_{i + 1}'] = nn.Linear(
                in_features=int(self.patch_size / (2 ** i)) * self.node_feat_dim,
                out_features=self.channel_embedding_dim, bias=True)
            self.projection_layer[f'edge_{i + 1}'] = nn.Linear(
                in_features=int(self.patch_size / (2 ** i)) * self.edge_feat_dim,
                out_features=self.channel_embedding_dim, bias=True)
            self.projection_layer[f'time_{i + 1}'] = nn.Linear(
                in_features=int(self.patch_size / (2 ** i)) * self.time_feat_dim,
                out_features=self.channel_embedding_dim, bias=True)
            self.projection_layer[f'neighbor_intersect_{i + 1}'] = nn.Linear(
                in_features=int(self.patch_size / (2 ** i)) * self.neighbor_intersect_feat_dim,
                out_features=self.channel_embedding_dim, bias=True)
            self.projection_layer[f'post_snapshot_{i + 1}'] = nn.Linear(
                in_features=int(self.patch_size / (2 ** i)) * self.snapshot_feat_dim,
                out_features=self.channel_embedding_dim, bias=True)
            self.projection_layer[f'snap_counts_{i + 1}'] = nn.Sequential(
                nn.Linear(in_features=int(self.patch_size / (2 ** i)) * self.num_snapshots,
                          out_features=self.channel_embedding_dim),
                nn.ReLU(),
                nn.Linear(in_features=self.channel_embedding_dim, out_features=self.channel_embedding_dim))

            self.transformers_dict[f'transformers_{i + 1}'] = nn.ModuleList([
                TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim,
                                   num_heads=self.num_heads, dropout=self.dropout)
                for _ in range(self.num_layers)
            ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim * self.num_patch_size,
                                      out_features=self.node_feat_dim, bias=True)

        self.feat_using = feat_using

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, snapshots: np.ndarray):
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list, src_nodes_neighbor_snapshots_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              snapshots=snapshots)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_nodes_neighbor_snapshots_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              snapshots=snapshots)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_padded_nodes_snapshots = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, node_snapshots=snapshots,
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               nodes_neighbor_snapshots_list=src_nodes_neighbor_snapshots_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_padded_nodes_snapshots = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, node_snapshots=snapshots,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               nodes_neighbor_snapshots_list=dst_nodes_neighbor_snapshots_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_intersect_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_intersect_feat_dim)
        # dst_padded_nodes_neighbor_intersect_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_intersect_feat_dim)
        if self.intersect_mode == 'sum':
            src_padded_nodes_neighbor_intersect_features, dst_padded_nodes_neighbor_intersect_features = \
                self.neighbor_intersect_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)
        else:
            src_padded_nodes_neighbor_intersect_features, dst_padded_nodes_neighbor_intersect_features = \
                self.neighbor_intersect_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                                                src_padded_nodes_snapshots=src_padded_nodes_snapshots,
                                                dst_padded_nodes_snapshots=dst_padded_nodes_snapshots,
                                                num_snapshots=self.num_snapshots)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, \
            src_padded_nodes_neighbor_snapshot_features, src_padded_nodes_neighbor_node_snap_counts = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder,
                              node_snapshots=snapshots, padded_nodes_neighbor_snapshots=src_padded_nodes_snapshots,
                              snapshot_encoder=self.snapshot_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, \
            dst_padded_nodes_neighbor_snapshot_features, dst_padded_nodes_neighbor_node_snap_counts = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder,
                              node_snapshots=snapshots, padded_nodes_neighbor_snapshots=dst_padded_nodes_snapshots,
                              snapshot_encoder=self.snapshot_encoder)

        src_patches_data_list, dst_patches_data_list = [], []

        for i in range(self.num_patch_size):
            # get the patches for source and destination nodes
            patch_size_local = int(self.patch_size / (2 ** i))
            src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
                src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_intersect_features, \
                src_patches_nodes_neighbor_snapshot_features, src_patches_nodes_neighbor_node_snap_counts = \
                self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                                 padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                                 padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                                 padded_nodes_neighbor_intersect_features=src_padded_nodes_neighbor_intersect_features,
                                 padded_nodes_neighbor_snapshot_features=src_padded_nodes_neighbor_snapshot_features,
                                 padded_nodes_neighbor_node_snap_counts=src_padded_nodes_neighbor_node_snap_counts,
                                 patch_size=patch_size_local)

            dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
                dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_intersect_features, \
                dst_patches_nodes_neighbor_snapshot_features, dst_patches_nodes_neighbor_node_snap_counts = \
                self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                                 padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                                 padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                                 padded_nodes_neighbor_intersect_features=dst_padded_nodes_neighbor_intersect_features,
                                 padded_nodes_neighbor_snapshot_features=dst_padded_nodes_neighbor_snapshot_features,
                                 padded_nodes_neighbor_node_snap_counts=dst_padded_nodes_neighbor_node_snap_counts,
                                 patch_size=patch_size_local)

            # align the patch encoding dimension
            # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
            src_patches_nodes_neighbor_node_raw_features = self.projection_layer[f'node_{i + 1}'](
                src_patches_nodes_neighbor_node_raw_features)
            src_patches_nodes_edge_raw_features = self.projection_layer[f'edge_{i + 1}'](
                src_patches_nodes_edge_raw_features)
            src_patches_nodes_neighbor_time_features = self.projection_layer[f'time_{i + 1}'](
                src_patches_nodes_neighbor_time_features)
            src_patches_nodes_neighbor_intersect_features = self.projection_layer[f'neighbor_intersect_{i + 1}'](
                src_patches_nodes_neighbor_intersect_features)
            src_patches_nodes_neighbor_snapshot_features = self.projection_layer[f'post_snapshot_{i + 1}'](
                src_patches_nodes_neighbor_snapshot_features)
            src_patches_nodes_neighbor_node_snap_counts = self.projection_layer[f'snap_counts_{i + 1}'](
                src_patches_nodes_neighbor_node_snap_counts)

            # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
            dst_patches_nodes_neighbor_node_raw_features = self.projection_layer[f'node_{i + 1}'](
                dst_patches_nodes_neighbor_node_raw_features)
            dst_patches_nodes_edge_raw_features = self.projection_layer[f'edge_{i + 1}'](
                dst_patches_nodes_edge_raw_features)
            dst_patches_nodes_neighbor_time_features = self.projection_layer[f'time_{i + 1}'](
                dst_patches_nodes_neighbor_time_features)
            dst_patches_nodes_neighbor_intersect_features = self.projection_layer[f'neighbor_intersect_{i + 1}'](
                dst_patches_nodes_neighbor_intersect_features)
            dst_patches_nodes_neighbor_snapshot_features = self.projection_layer[f'post_snapshot_{i + 1}'](
                dst_patches_nodes_neighbor_snapshot_features)
            dst_patches_nodes_neighbor_node_snap_counts = self.projection_layer[f'snap_counts_{i + 1}'](
                dst_patches_nodes_neighbor_node_snap_counts)

            batch_size = len(src_patches_nodes_neighbor_node_raw_features)
            src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
            dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
            patches_nodes_neighbor_node_raw_features = torch.cat(
                [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
            patches_nodes_edge_raw_features = torch.cat(
                [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
            patches_nodes_neighbor_time_features = torch.cat(
                [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
            patches_nodes_neighbor_intersect_features = torch.cat(
                [src_patches_nodes_neighbor_intersect_features, dst_patches_nodes_neighbor_intersect_features], dim=1)

            patches_nodes_neighbor_snapshot_features = torch.cat(
                [src_patches_nodes_neighbor_snapshot_features, dst_patches_nodes_neighbor_snapshot_features], dim=1)

            patches_nodes_neighbor_node_snap_counts = torch.cat(
                [src_patches_nodes_neighbor_node_snap_counts, dst_patches_nodes_neighbor_node_snap_counts], dim=1)

            patches_dict = {'time_feat': patches_nodes_neighbor_time_features,
                            'intersect_feat': patches_nodes_neighbor_intersect_features,
                            'snapshot_feat': patches_nodes_neighbor_snapshot_features,
                            'snap_counts': patches_nodes_neighbor_node_snap_counts}

            patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features]

            for j in self.feat_using:
                patches_data.append(patches_dict[j])

            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
            patches_data = torch.stack(patches_data, dim=2)
            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
            patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches,
                                                self.num_channels * self.channel_embedding_dim)

            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
            for transformer in self.transformers_dict[f'transformers_{i + 1}']:
                patches_data = transformer(patches_data)

            # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            src_patches_data = patches_data[:, : src_num_patches, :]
            # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
            dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
            # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            src_patches_data = torch.mean(src_patches_data, dim=1)
            # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            dst_patches_data = torch.mean(dst_patches_data, dim=1)

            src_patches_data_list.append(src_patches_data)
            dst_patches_data_list.append(dst_patches_data)

        src_patches_data = torch.cat(src_patches_data_list, dim=1)
        dst_patches_data = torch.cat(dst_patches_data_list, dim=1)
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, node_snapshots: np.array,
                      nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, nodes_neighbor_snapshots_list: list, patch_size: int = 1,
                      max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(
                nodes_neighbor_times_list[idx]) == len(nodes_neighbor_snapshots_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_snapshots_list[idx] = nodes_neighbor_snapshots_list[idx][
                                                     -(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)
        padded_nodes_neighbor_snapshots = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            padded_nodes_neighbor_snapshots[idx, 0] = node_snapshots[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = \
                nodes_neighbor_times_list[idx]
                padded_nodes_neighbor_snapshots[idx, 1: len(nodes_neighbor_snapshots_list[idx]) + 1] = \
                nodes_neighbor_snapshots_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, padded_nodes_neighbor_snapshots

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder,
                     node_snapshots: np.ndarray, padded_nodes_neighbor_snapshots: np.ndarray,
                     snapshot_encoder: PositionalEncoding1D):
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]

        # Tensor, shape (batch_size, max_seq_length, num_snapshots)
        padded_nodes_neighbor_node_snap_counts = self.node_snap_counts[torch.from_numpy(padded_nodes_neighbor_ids)]

        batch_size = padded_nodes_neighbor_node_snap_counts.shape[0]
        max_seq_length = padded_nodes_neighbor_node_snap_counts.shape[1]

        node_snapshots_tensor = torch.tensor(node_snapshots)

        mask = torch.arange(self.num_snapshots).expand(batch_size, max_seq_length, self.num_snapshots) < (
                                                                                                                     node_snapshots_tensor - 1)[
                                                                                                         :, None, None]
        masked_padded_nodes_neighbor_node_snap_counts = padded_nodes_neighbor_node_snap_counts.clone()
        masked_padded_nodes_neighbor_node_snap_counts[~mask] = 0

        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(
                self.device))

        padded_nodes_neighbor_snapshot_features = snapshot_encoder(
            timestamps=torch.from_numpy(node_snapshots[:, np.newaxis] - padded_nodes_neighbor_snapshots).float().to(
                self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0
        padded_nodes_neighbor_snapshot_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features, padded_nodes_neighbor_snapshot_features, masked_padded_nodes_neighbor_node_snap_counts

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor,
                    padded_nodes_neighbor_intersect_features: torch.Tensor = None,
                    padded_nodes_neighbor_snapshot_features: torch.Tensor = None,
                    padded_nodes_neighbor_node_snap_counts: torch.Tensor = None,
                    patch_size: int = 1):

        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
            patches_nodes_neighbor_time_features, patches_nodes_neighbor_intersect_features, \
            patches_nodes_neighbor_snapshot_features, patches_nodes_neighbor_node_snap_counts = [], [], [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_snapshot_features.append(
                padded_nodes_neighbor_snapshot_features[:, start_idx: end_idx, :])
            if padded_nodes_neighbor_intersect_features is not None:
                patches_nodes_neighbor_intersect_features.append(
                    padded_nodes_neighbor_intersect_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_node_snap_counts.append(
                padded_nodes_neighbor_node_snap_counts[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size,
                                                                                                      num_patches,
                                                                                                      patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_snapshot_features = torch.stack(patches_nodes_neighbor_snapshot_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.snapshot_feat_dim)

        if padded_nodes_neighbor_intersect_features is not None:
            patches_nodes_neighbor_intersect_features = torch.stack(patches_nodes_neighbor_intersect_features,
                                                                    dim=1).reshape(batch_size, num_patches,
                                                                                   patch_size * self.neighbor_intersect_feat_dim)

        patches_nodes_neighbor_node_snap_counts = torch.stack(patches_nodes_neighbor_node_snap_counts, dim=1).reshape(
            batch_size, num_patches, patch_size * self.num_snapshots)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_intersect_features, patches_nodes_neighbor_snapshot_features, patches_nodes_neighbor_node_snap_counts

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborIntersectEncoder(nn.Module):

    def __init__(self, neighbor_intersect_feat_dim: int, device: str = 'cpu'):
        super(NeighborIntersectEncoder, self).__init__()
        self.neighbor_intersect_feat_dim = neighbor_intersect_feat_dim
        self.device = device

        self.neighbor_intersect_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_intersect_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_intersect_feat_dim, out_features=self.neighbor_intersect_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray):
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids,
                                                                              dst_padded_nodes_neighbor_ids):
            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_intersect_feat_dim)
        src_padded_nodes_neighbor_intersect_features = self.neighbor_intersect_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_intersect_feat_dim)
        dst_padded_nodes_neighbor_intersect_features = self.neighbor_intersect_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_intersect_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_intersect_feat_dim)
        # dst_padded_nodes_neighbor_intersect_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_intersect_feat_dim)
        return src_padded_nodes_neighbor_intersect_features, dst_padded_nodes_neighbor_intersect_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = \
        self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(
            0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs


class AppearancesEncoder(nn.Module):

    def __init__(self, neighbor_intersect_feat_dim: int, num_snapshots: int, intersect_mode: str, device: str = 'cpu'):
        super(AppearancesEncoder, self).__init__()
        self.neighbor_intersect_feat_dim = neighbor_intersect_feat_dim
        self.device = device
        self.num_snapshots = num_snapshots

        self.neighbor_intersect_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_intersect_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_intersect_feat_dim, out_features=self.neighbor_intersect_feat_dim))

        self.intersect_mode = intersect_mode
        if self.intersect_mode == "mlp":
            self.agg_snapshots = AggregateSnapshots(self.num_snapshots, self.neighbor_intersect_feat_dim)
        elif self.intersect_mode == "gru":
            self.agg_snapshots = SnapshotGRU(2, self.neighbor_intersect_feat_dim)
        else:
            raise ValueError("some thing wrong with intersect_mode")

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray,
                                src_padded_nodes_snapshots: np.ndarray,
                                dst_padded_nodes_snapshots: np.ndarray,
                                num_snapshots):
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for (src_ids, dst_ids, src_snaps, dst_snaps) in zip(src_padded_nodes_neighbor_ids,
                                                            dst_padded_nodes_neighbor_ids,
                                                            src_padded_nodes_snapshots, dst_padded_nodes_snapshots):

            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_ids, return_inverse=True,
                                                                         return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_ids, return_inverse=True,
                                                                         return_counts=True)

            # shape (src_max_seq_length, max_snapshot_id, 2) 和 (dst_max_seq_length, max_snapshot_id, 2)
            src_snapshot_appearances = torch.zeros((len(src_ids), num_snapshots, 2), dtype=torch.float32,
                                                   device=self.device)
            dst_snapshot_appearances = torch.zeros((len(dst_ids), num_snapshots, 2), dtype=torch.float32,
                                                   device=self.device)
            for idx, node_id in enumerate(src_ids):
                if node_id == 0:
                    continue
                src_snapshot_id = src_snaps[idx]
                if src_snapshot_id == dst_snaps[idx]:  # 检查是否在相同的snapshot
                    src_count = src_counts[src_inverse_indices[idx]]
                    dst_count = dst_counts[dst_inverse_indices[idx]] if node_id in dst_unique_keys else 0
                    src_snapshot_appearances[idx, int(src_snapshot_id), 0] += src_count
                    src_snapshot_appearances[idx, int(src_snapshot_id), 1] += dst_count

            for idx, node_id in enumerate(dst_ids):
                if node_id == 0:
                    continue
                dst_snapshot_id = dst_snaps[idx]
                if dst_snapshot_id == src_snaps[idx]:
                    dst_count = dst_counts[dst_inverse_indices[idx]]
                    src_count = src_counts[src_inverse_indices[idx]] if node_id in src_unique_keys else 0
                    dst_snapshot_appearances[idx, int(dst_snapshot_id), 0] += src_count
                    dst_snapshot_appearances[idx, int(dst_snapshot_id), 1] += dst_count

            src_padded_nodes_appearances.append(src_snapshot_appearances)
            dst_padded_nodes_appearances.append(dst_snapshot_appearances)

        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def count_intersect_in_snapshots(self, src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids,
                                     src_padded_nodes_snapshots, dst_padded_nodes_snapshots, num_snapshots):

        batch_size = src_padded_nodes_neighbor_ids.shape[0]
        src_max_seq_length = src_padded_nodes_neighbor_ids.shape[1]
        dst_max_seq_length = dst_padded_nodes_neighbor_ids.shape[1]

        src_padded_nodes_intersects = torch.zeros(batch_size, src_max_seq_length, 2, num_snapshots,
                                                  dtype=torch.float32)
        dst_padded_nodes_intersects = torch.zeros(batch_size, dst_max_seq_length, 2, num_snapshots,
                                                  dtype=torch.float32)

        for i in range(batch_size):
            for src_index in range(src_max_seq_length):
                src_neighbor_id = src_padded_nodes_neighbor_ids[i, src_index]
                src_snapshot_id = int(src_padded_nodes_snapshots[i, src_index]) - 1
                if src_neighbor_id != 0:
                    self_mask = (src_padded_nodes_neighbor_ids[i] == src_neighbor_id) & (
                            src_padded_nodes_snapshots[i] == src_snapshot_id + 1)
                    src_padded_nodes_intersects[i, src_index, 0, src_snapshot_id] = self_mask.astype(float).sum()

                    other_mask = (dst_padded_nodes_neighbor_ids[i] == src_neighbor_id) & (
                            dst_padded_nodes_snapshots[i] == src_snapshot_id + 1)
                    src_padded_nodes_intersects[i, src_index, 1, src_snapshot_id] = other_mask.astype(float).sum()

            for dst_index in range(dst_max_seq_length):
                dst_neighbor_id = dst_padded_nodes_neighbor_ids[i, dst_index]
                dst_snapshot_id = int(dst_padded_nodes_snapshots[i, dst_index]) - 1
                if dst_neighbor_id != 0:
                    self_mask = (dst_padded_nodes_neighbor_ids[i] == dst_neighbor_id) & (
                            dst_padded_nodes_snapshots[i] == dst_snapshot_id + 1)
                    dst_padded_nodes_intersects[i, dst_index, 1, dst_snapshot_id] = self_mask.astype(float).sum()

                    other_mask = (src_padded_nodes_neighbor_ids[i] == dst_neighbor_id) & (
                            src_padded_nodes_snapshots[i] == dst_snapshot_id + 1)
                    dst_padded_nodes_intersects[i, dst_index, 0, dst_snapshot_id] = other_mask.astype(float).sum()

        return src_padded_nodes_intersects, dst_padded_nodes_intersects

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray,
                src_padded_nodes_snapshots: np.ndarray, dst_padded_nodes_snapshots: np.ndarray, num_snapshots):
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_intersect_in_snapshots(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            src_padded_nodes_snapshots=src_padded_nodes_snapshots,
            dst_padded_nodes_snapshots=dst_padded_nodes_snapshots,
            num_snapshots=num_snapshots)
        src_padded_nodes_appearances = src_padded_nodes_appearances.to(self.device)
        dst_padded_nodes_appearances = dst_padded_nodes_appearances.to(self.device)

        if self.intersect_mode == 'mlp':
            src_padded_nodes_neighbor_snap_features = self.agg_snapshots(src_padded_nodes_appearances)
            dst_padded_nodes_neighbor_snap_features = self.agg_snapshots(dst_padded_nodes_appearances)

            src_padded_nodes_neighbor_intersect_features = self.neighbor_intersect_encode_layer(
                src_padded_nodes_neighbor_snap_features.unsqueeze(dim=-1)).sum(dim=2)
            dst_padded_nodes_neighbor_intersect_features = self.neighbor_intersect_encode_layer(
                dst_padded_nodes_neighbor_snap_features.unsqueeze(dim=-1)).sum(dim=2)
        elif self.intersect_mode == 'gru':
            _, seq_length_src, _, _ = src_padded_nodes_appearances.shape
            _, seq_length_dst, _, _ = dst_padded_nodes_appearances.shape
            src_padded_nodes_appearances = src_padded_nodes_appearances.permute(1, 0, 3, 2)
            dst_padded_nodes_appearances = dst_padded_nodes_appearances.permute(1, 0, 3, 2)

            output_list_src = []
            output_list_dst = []

            for t in range(seq_length_src):
                snapshot_input_src = src_padded_nodes_appearances[t]
                gru_output = self.agg_snapshots(snapshot_input_src)
                output_list_src.append(gru_output.unsqueeze(1))

            for t in range(seq_length_dst):
                snapshot_input_dst = dst_padded_nodes_appearances[t]
                gru_output = self.agg_snapshots(snapshot_input_dst)
                output_list_dst.append(gru_output.unsqueeze(1))

            src_padded_nodes_neighbor_intersect_features = torch.cat(output_list_src, dim=1)
            dst_padded_nodes_neighbor_intersect_features = torch.cat(output_list_dst, dim=1)
        else:
            raise NotImplementedError

        return src_padded_nodes_neighbor_intersect_features, dst_padded_nodes_neighbor_intersect_features


class AggregateSnapshots(nn.Module):
    def __init__(self, num_snapshots, hidden_size):
        super(AggregateSnapshots, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_snapshots, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        batch_size, src_max_seq_length, _, num_snapshots = x.shape
        x = x.view(-1, num_snapshots)
        x = self.mlp(x)
        x = x.view(batch_size, src_max_seq_length, 2, 2)
        x = x.mean(dim=2)
        return x


class SnapshotGRU(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers=1):
        super(SnapshotGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)
        return output[:, -1, :]
