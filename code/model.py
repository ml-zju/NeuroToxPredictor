"""Multi-modal gated NeuroToxPredictor architecture."""

from __future__ import annotations

import dgl
import numpy as np
import torch
from dgl.nn.pytorch import GATConv
from torch import nn
from torch.nn import functional as F


class GraphBranch(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, heads=2, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(
                in_dim,
                hidden_dims[0],
                heads,
                feat_drop=dropout,
                attn_drop=dropout,
                allow_zero_in_degree=True,
            )
        )
        for index in range(len(hidden_dims) - 1):
            self.layers.append(
                GATConv(
                    hidden_dims[index],
                    hidden_dims[index + 1],
                    heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True,
                )
            )
        self.projection = nn.Linear(int(np.sum(hidden_dims)), out_dim)

    def forward(self, graph):
        features = graph.ndata["x"]
        pooled = []
        for layer in self.layers:
            features = self.dropout(F.relu(layer(graph, features).mean(dim=1)))
            graph.ndata["h"] = features
            pooled.append(dgl.mean_nodes(graph, "h"))
        return self.projection(self.dropout(torch.cat(pooled, dim=1)))


class FingerBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        features = self.dropout(F.relu(self.linear1(features)))
        return self.dropout(F.relu(self.linear2(features)))


class ChemBERTaBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        features = self.dropout(F.relu(self.norm1(self.linear1(features))))
        return self.dropout(F.relu(self.norm2(self.linear2(features))))


class GatedNeuroToxPredictor(nn.Module):
    def __init__(
        self,
        graph_in_dim,
        graph_hidden_dims,
        graph_out_dim,
        finger_in_dim,
        finger_hidden_dim,
        finger_out_dim,
        chem_in_dim,
        chem_hidden_dim,
        chem_out_dim,
        heads=2,
        dropout=0.5,
        anti_collapse=False,
        gate_projection_dim=32,
        min_gate_weight=0.05,
    ):
        super().__init__()
        self.anti_collapse = anti_collapse
        self.min_gate_weight = min_gate_weight
        self.graph = GraphBranch(
            graph_in_dim, graph_hidden_dims, graph_out_dim, heads, dropout
        )
        self.finger = FingerBranch(
            finger_in_dim, finger_hidden_dim, finger_out_dim, dropout
        )
        self.chemberta = ChemBERTaBranch(
            chem_in_dim, chem_hidden_dim, chem_out_dim, dropout
        )
        fusion_dim = graph_out_dim + finger_out_dim + chem_out_dim

        if anti_collapse:
            self.graph_gate_projection = nn.Sequential(
                nn.Linear(graph_out_dim, gate_projection_dim),
                nn.LayerNorm(gate_projection_dim),
            )
            self.finger_gate_projection = nn.Sequential(
                nn.Linear(finger_out_dim, gate_projection_dim),
                nn.LayerNorm(gate_projection_dim),
            )
            self.chem_gate_projection = nn.Sequential(
                nn.Linear(chem_out_dim, gate_projection_dim),
                nn.LayerNorm(gate_projection_dim),
            )
            self.gate = nn.Linear(gate_projection_dim * 3, 3)
            self.graph_aux_head = nn.Linear(graph_out_dim, 2)
            self.finger_aux_head = nn.Linear(finger_out_dim, 2)
            self.chem_aux_head = nn.Linear(chem_out_dim, 2)
        else:
            self.gate = nn.Linear(fusion_dim, 3)

        self.classifier = nn.Linear(fusion_dim, 2)

    def forward(
        self,
        graph,
        fingerprints,
        chemberta,
        temperature=1.0,
        force_uniform=False,
    ):
        graph_output = self.graph(graph)
        finger_output = self.finger(fingerprints)
        chem_output = self.chemberta(chemberta)

        if self.anti_collapse:
            gate_input = torch.cat(
                (
                    self.graph_gate_projection(graph_output),
                    self.finger_gate_projection(finger_output),
                    self.chem_gate_projection(chem_output),
                ),
                dim=1,
            )
        else:
            gate_input = torch.cat(
                (graph_output, finger_output, chem_output), dim=1
            )

        gate_logits = self.gate(gate_input)
        if force_uniform:
            weights = torch.full_like(gate_logits, 1.0 / 3.0)
        else:
            weights = torch.softmax(gate_logits / temperature, dim=1)
            if self.anti_collapse and self.min_gate_weight > 0:
                weights = (
                    weights * (1.0 - 3.0 * self.min_gate_weight)
                    + self.min_gate_weight
                )

        fused = torch.cat(
            (
                graph_output * weights[:, 0:1],
                finger_output * weights[:, 1:2],
                chem_output * weights[:, 2:3],
            ),
            dim=1,
        )
        output = {
            "logits": self.classifier(fused),
            "weights": weights,
            "branch_outputs": (graph_output, finger_output, chem_output),
        }
        if self.anti_collapse:
            output["aux_logits"] = (
                self.graph_aux_head(graph_output),
                self.finger_aux_head(finger_output),
                self.chem_aux_head(chem_output),
            )
        return output
