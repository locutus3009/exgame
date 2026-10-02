// SPDX-License-Identifier: MIT

#version 460

struct Row {
    uint c[128];
};

struct Incidence {
    uint row;
};

struct Fatals {
    float fatal[1];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params { uint count; };

layout(set = 0, binding = 0) buffer FatalsData {
    Fatals data[];
} fatals;

layout(set = 0, binding = 1) buffer Index {
    Incidence data[];
} index;

layout(set = 0, binding = 2) buffer RowData {
    Row data[];
} rows;

// The flat scalar map. Row words 0/1/2 are the a/b/out slots into it.
layout(set = 0, binding = 3) buffer Data {
    float data[];
} buf;


void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= count) return;

    uint r = index.data[idx].row;
    uint idx_a = rows.data[r].c[0];
    uint idx_b = rows.data[r].c[1];
    uint idx_out = rows.data[r].c[2];

    buf.data[idx_out] = buf.data[idx_a] + buf.data[idx_b];
    fatals.data[idx].fatal[0] = 0.0;
}
