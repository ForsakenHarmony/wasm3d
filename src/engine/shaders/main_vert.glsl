#version 300 es

#{{defs}}

layout(location=0) in vec4 a_position;
layout(location=1) in vec2 a_texcoord;
layout(location=2) in vec3 a_normal;
layout(location=3) in vec4 a_color;

uniform mat4 u_mvp;

#ifdef TEXTURE
out vec2 v_texcoord;
#endif
#ifdef COLOR
out vec4 v_color;
#endif

void main() {
  gl_Position = u_mvp * a_position;

  #ifdef TEXTURE
  v_texcoord = a_texcoord;
  #endif
  #ifdef COLOR
  v_color = a_color;
  #endif
}
