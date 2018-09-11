#version 300 es

#{{defs}}

precision mediump float;

#ifdef COLOR
in vec4 v_color;
#endif
#ifdef TEXTURE
in vec2 v_texcoord;
#endif

#ifdef TEXTURE
uniform sampler2D u_texture;
#endif

out vec4 outColor;

void main() {
  #ifdef TEXTURE
  outColor = texture(u_texture, v_texcoord);
  #endif
  #ifdef COLOR
  outColor = vec4(0,1,0,1);
  #endif
}
