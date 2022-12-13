#include <metal_stdlib>
using namespace metal;



/***********************************************************************************************
* Vertex Shader
***********************************************************************************************/


struct VertexIn {
    float4 position [[ attribute(0) ]];
    float4 color [[ attribute(1) ]];
};

struct VertexOut {
    float4 position [[ position ]];
    float4 color;
};

vertex VertexOut vertex_main(VertexIn in [[ stage_in ]] ,
                             constant float4x4 &projectionMatrix [[ buffer(1) ]],
                             constant float4x4 &viewMatrix [[ buffer(2) ]]) {
    VertexOut out;
    out.position = in.position * projectionMatrix * viewMatrix;
    out.color = in.color;
    return out;
}


/*****************************************************************************************
 * Fragment Shader
 *****************************************************************************************/

struct FragmentIn {
    float4 position [[ position ]];
    float4 color;
};

struct FragmentOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment FragmentOut fragment_main(FragmentIn in [[ stage_in ]]) {
    FragmentOut out;
    float4 color = in.color;
    float depth = in.position.z;
    color.a = depth;
    out.color = color;
    out.depth = in.position.z;
    return out;
}

