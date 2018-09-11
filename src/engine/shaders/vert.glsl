#version 300 es

#{{defs}}

in vec4 a_position;

#ifdef TEXTURE
in vec2 a_texcoord;
#endif
#ifdef COLOR
in vec4 a_color;
#endif

uniform mat4 u_matrix;

#ifdef TEXTURE
out vec2 v_texcoord;
#endif
#ifdef COLOR
out vec4 v_color;
#endif

void main() {
  gl_Position = u_matrix * a_position;

  #ifdef TEXTURE
  v_texcoord = a_texcoord;
  #endif
  #ifdef COLOR
  v_color = a_color;
  #endif
}
