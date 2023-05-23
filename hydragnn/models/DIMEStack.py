"""
DimeNet
========
Directional message passing neural network
for molecular graphs. The convolutional
layer uses spherical and radial basis
functions to perform message passing.

In particular this message passing layer
relies on the angle formed by the triplet
of incomming and outgoing messages.

The three key components of this network are
outlined below. In particular, the convolutional
network that is used for the message passing
the triplet function that generates to/from
information for angular values, and finally
the radial basis embedding that is used to
include radial basis information.

"""
from typing import Callable, Tuple
from torch_geometric.typing import SparseTensor

import torch
from torch import Tensor
from torch.nn import SiLU

from torch_geometric.nn import Linear, Sequential
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    InteractionBlock,
    SphericalBasisLayer,
    OutputBlock,
)
from .Base import Base


class DIMEStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        num_bilinear,
        num_radial,
        num_spherical,
        radius,
        envelope_exponent,
        num_before_skip,
        num_after_skip,
        *args,
        **kwargs
    ):
        self.num_bilinear = num_bilinear
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.radius = radius

        super().__init__(*args, **kwargs)

        self.rbf = BesselBasisLayer(num_radial, radius, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, radius, envelope_exponent
        )

        
        pass


    def get_conv(self, input_dim, output_dim):
        emb = EmbeddingBlock(self.num_radial, input_dim, act=SiLU())
        inter = InteractionBlock(
            hidden_channels=self.hidden_dim,
            num_bilinear=self.num_bilinear,
            num_spherical=self.num_spherical,
            num_radial=self.num_radial,
            num_before_skip=self.num_before_skip,
            num_after_skip=self.num_after_skip,
            act=SiLU(),
            )
        dec = OutputBlock(self.num_radial, self.hidden_dim, output_dim, 1, SiLU())
        return Sequential('x, rbf, sbf, i, j, idx_kj, idx_ji', [
            (emb, 'x, rbf, i, j -> x1'),
            (inter,'x1, rbf, sbf, idx_kj, idx_ji -> x2'),
            (dec,'x2, rbf, i -> c'),
        ])

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "DimeNet requires node positions (data.pos) to be set."
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            data.edge_index, num_nodes=data.x.size(0)
        )
        dist = (data.pos[i] - data.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = data.pos[idx_i]
        pos_ji, pos_ki = data.pos[idx_j] - pos_i, data.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        conv_args = {"rbf":rbf, "sbf":sbf, "i": i, "j":j, "idx_kj":idx_kj, "idx_ji":idx_ji}

        return conv_args


"""
Triplets
---------
Generates to/from edge_indices for
angle generating purposes.

"""


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        # self.emb = Embedding(95, hidden_channels) # Atomic Embeddings are handles by Hydra
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        # x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))