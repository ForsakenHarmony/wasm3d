pub mod glenum;

pub use glenum::*;

use std::ops::Deref;
use stdweb::unstable::{TryFrom, TryInto};
use stdweb::web::html_element::CanvasElement;
use stdweb::web::*;
use stdweb::InstanceOf;
use stdweb::Reference;
use stdweb::UnsafeTypedArray;

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGL2RenderingContext")]
pub struct WebGL2RenderingContext(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLBuffer")]
pub struct WebGLBuffer(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLShader")]
pub struct WebGLShader(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLProgram")]
pub struct WebGLProgram(Reference);

#[derive(Debug)]
pub struct WebGLActiveInfo {
  reference: Reference,
  name: String,
  size: u32,
  kind: UniformType,
}

impl WebGLActiveInfo {
  pub fn new<T: Into<String>>(
    name: T,
    size: u32,
    kind: UniformType,
    reference: Reference,
  ) -> WebGLActiveInfo {
    let nam = name.into();
    WebGLActiveInfo {
      reference,
      name: nam,
      size,
      kind,
    }
  }
}

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLTexture")]
pub struct WebGLTexture(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLVertexArrayObject")]
pub struct WebGLVertexArrayObject(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLUniformLocation")]
pub struct WebGLUniformLocation(Reference);

#[derive(Debug, Clone, ReferenceType)]
#[reference(instance_of = "WebGLFramebuffer")]
pub struct WebGLFramebuffer(Reference);

impl WebGL2RenderingContext {
  pub fn log<T: Into<String>>(&self, msg: T) {
    js! { console.log(@{msg.into()})};
  }

  pub fn new(canvas: &CanvasElement) -> Self {
    use stdweb;
    let gl = js! {
      const gl = (@{canvas}).getContext("webgl2");
      console.log("opengl", gl.getParameter(gl.VERSION));
      console.log("shading language", gl.getParameter(gl.SHADING_LANGUAGE_VERSION));
      console.log("vendor", gl.getParameter(gl.VENDOR));
      return gl;
    };
    WebGL2RenderingContext(gl.try_into().unwrap())
  }

  pub fn create_buffer(&self) -> WebGLBuffer {
    let value = js! {
      return @{self}.createBuffer();
    };
    WebGLBuffer(value.try_into().expect("error: create_buffer"))
  }

  pub fn delete_buffer(&self, buffer: &WebGLBuffer) {
    js! {@(no_return)
      (@{self}).deleteBuffer(@{buffer.deref()});
    }
  }

  pub fn buffer_data_bytes(&self, kind: BufferKind, data: &[u8], draw: DrawMode) {
    js! {@(no_return)
      (@{&self}).bufferData(@{kind as u32},@{ TypedArray::from(data) }, @{draw as u32});
    };
  }

  pub fn buffer_data(&self, kind: BufferKind, data: &[f32], draw: DrawMode) {
    js! {@(no_return)
      (@{&self}).bufferData(@{kind as u32},@{ TypedArray::from(data) }, @{draw as u32});
    };
  }

  pub fn buffer_sub_data(&self, kind: BufferKind, offset: u32, data: &[u8]) {
    js! {@(no_return)
      (@{&self}).bufferSubData(@{kind as u32},@{offset},@{ unsafe { UnsafeTypedArray::new(data) } });
    };
  }

  pub fn bind_buffer(&self, kind: BufferKind, buffer: &WebGLBuffer) {
    js! {@(no_return)
      (@{self}).bindBuffer(@{kind as u32},@{buffer.deref()});
    };
  }

  pub fn unbind_buffer(&self, kind: BufferKind) {
    js! {@(no_return)
      (@{self}).bindBuffer(@{kind as u32},null);
    };
  }

  pub fn create_shader(&self, kind: ShaderKind) -> WebGLShader {
    let value = js! {
      return (@{self}).createShader(@{ kind as u32 });
    };
    WebGLShader(value.try_into().unwrap())
  }

  pub fn shader_source(&self, shader: &WebGLShader, code: &str) {
    js! {@(no_return)
      (@{self}).shaderSource(@{shader.deref()},@{ code });
    };
  }

  pub fn compile_shader(&self, shader: &WebGLShader) {
    js! {@(no_return)
      (@{self}).compileShader(@{shader.deref()});
      const compiled = (@{self}).getShaderParameter(@{shader.deref()}, 0x8B81);
      console.log("Shader compiled successfully:", compiled);
      if (!compiled) {
        const compilationLog = (@{self}).getShaderInfoLog(@{shader.deref()});
        console.log("Shader compiler log:",compilationLog);
      }
    };
  }

  pub fn create_program(&self) -> WebGLProgram {
    let value = js! {
      return (@{self}).createProgram();
    };
    WebGLProgram(value.try_into().unwrap())
  }

  pub fn link_program(&self, program: &WebGLProgram) {
    js! {@(no_return)
      (@{self}).linkProgram(@{program.deref()});
      const linked = (@{self}).getProgramParameter(@{program.deref()}, 35714);
      console.log("Program linked successfully:", linked);
      if (!linked) {
        const compilationLog = (@{self}).getProgramInfoLog(@{program.deref()});
        console.log("Program linking log:",compilationLog);
      }
    };
  }

  pub fn use_program(&self, program: &WebGLProgram) {
    js! {@(no_return)
      (@{self}).useProgram(@{program.deref()});
    };
  }

  pub fn attach_shader(&self, program: &WebGLProgram, shader: &WebGLShader) {
    js! {@(no_return)
      (@{self}).attachShader(@{program.deref()},@{shader.deref()})
    };
  }

  pub fn get_attrib_location(&self, program: &WebGLProgram, name: &str) -> Option<u32> {
    let value = js! {
      return (@{self}).getAttribLocation(@{program.deref()},@{name})
    };
    value.try_into().ok()
  }

  pub fn get_uniform_location(
    &self,
    program: &WebGLProgram,
    name: &str,
  ) -> Option<WebGLUniformLocation> {
    let value = js! {
      const res = (@{self}).getUniformLocation(@{program.deref()},@{name});
      return res;
    };
    value.try_into().ok().map(WebGLUniformLocation)
  }

  pub fn vertex_attrib_pointer(
    &self,
    location: u32,
    size: AttributeSize,
    kind: DataType,
    normalized: bool,
    stride: u32,
    offset: u32,
  ) {
    js! {
      (@{self}).vertexAttribPointer(@{location},@{size as u16},@{kind as i32},@{normalized},@{stride},@{offset});
    };
  }

  pub fn enable_vertex_attrib_array(&self, location: u32) {
    js! {@(no_return)
      @{self}.enableVertexAttribArray(@{location});
    };
  }

  pub fn clear_color(&self, r: f32, g: f32, b: f32, a: f32) {
    js! {@(no_return)
      @{self}.clearColor(@{r},@{g},@{b},@{a})
    };
  }

  pub fn enable(&self, flag: Flag) {
    js! {@(no_return)
      (@{self}).enable(@{flag as i32});
    };
  }

  pub fn clear(&self, bit: BufferBit) {
    js! {@(no_return)
      (@{self}).clear(@{bit as i32});
    };
  }

  pub fn viewport(&self, x: i32, y: i32, width: u32, height: u32) {
    js! {@(no_return)
      (@{self}).viewport(@{x},@{y},@{width},@{height});
    };
  }

  pub fn draw_elements(&self, mode: Primitives, count: usize, kind: DataType, offset: u32) {
    js! {@(no_return)
      (@{self}).drawElements(@{mode as i32},@{count as u32},@{kind as i32},@{offset});
    };
  }

  pub fn draw_arrays(&self, mode: Primitives, count: usize) {
    js! {@(no_return)
      (@{self}).drawArrays(@{mode as i32},0,@{count as i32});
    };
  }

  pub fn pixel_storei(&self, storage: PixelStorageMode, value: i32) {
    js! {@(no_return)
      (@{self}).pixelStorei(@{storage as i32},@{value});
    }
  }

  pub fn tex_image2d(
    &self,
    target: TextureBindPoint,
    level: u8,
    width: u16,
    height: u16,
    format: PixelFormat,
    kind: DataType,
    pixels: &[u8],
  ) {
    js! {@(no_return)
      @{self}.texImage2D(
        @{target as u32},@{level as u32},@{format as u32},
        @{width as u32},@{height as u32},0,
        @{format as u32},@{kind as u32},
        @{unsafe { UnsafeTypedArray::new(pixels)} }
      );
    };
  }

  pub fn tex_sub_image2d(
    &self,
    target: TextureBindPoint,
    level: u8,
    xoffset: u16,
    yoffset: u16,
    width: u16,
    height: u16,
    format: PixelFormat,
    kind: DataType,
    pixels: &[u8],
  ) {
    js! {
      (@{self}).texSubImage2D(
        @{target as u32},@{level as u32},@{xoffset as u32},@{yoffset as u32},
        @{width as u32},@{height as u32},@{format as u32},@{kind as u32},
        @{unsafe {UnsafeTypedArray::new(pixels)}}
      );
    };
  }

  pub fn compressed_tex_image2d(
    &self,
    target: TextureBindPoint,
    level: u8,
    compression: TextureCompression,
    width: u16,
    height: u16,
    data: &[u8],
  ) {
    js! {@(no_return)
      (@{self}).getExtension("WEBGL_compressed_texture_s3tc") ||
      (@{self}).getExtension("MOZ_WEBGL_compressed_texture_s3tc") ||
      (@{self}).getExtension("WEBKIT_WEBGL_compressed_texture_s3tc")
      (@{self}).compressedTexImage2D(
        @{target as u32},
        @{level as u32},
        @{compression as u16},
        @{width as u32},
        @{height as u32},
        0,
        @{unsafe { UnsafeTypedArray::new(data) }}
      );
    }
  }

  ///
  pub fn create_texture(&self) -> WebGLTexture {
    let handle = js! {
      return @{self}.createTexture()
    };
    WebGLTexture(handle.try_into().unwrap())
  }

  pub fn delete_texture(&self, texture: &WebGLTexture) {
    js! {@(no_return)
      (@{self}).deleteTexture(@{&texture.0});
    }
  }

  pub fn bind_texture(&self, texture: &WebGLTexture) {
    js! {@(no_return)
      (@{self}).bindTexture(@{TextureBindPoint::Texture2d as u32 }, @{&texture.0})
    }
  }

  pub fn active_texture(&self, texture: TextureIndex) {
    js! {@(no_return)
      (@{self}).activeTexture(@{texture as u32})
    }
  }

  pub fn unbind_texture(&self) {
    js! {@(no_return)
      (@{self}).bindTexture(@{TextureBindPoint::Texture2d as u32 },null)
    }
  }

  pub fn generate_mipmap(&self, target: TextureKind) {
    js! {@(no_return)
      (@{self}).generateMipmap(@{target as u32})
    }
  }

  pub fn blend_func(&self, b1: BlendMode, b2: BlendMode) {
    js! {@(no_return)
      (@{self}).blendFunc(@{b1 as u32},@{b2 as u32})
    }
  }

  pub fn blend_color(&self, r: f32, g: f32, b: f32, a: f32) {
    js! {@(no_return)
      @{self}.blendColor(@{r}, @{g}, @{b}, @{a});
    }
  }

  pub fn uniform_matrix_4fv(&self, location: &WebGLUniformLocation, value: &[[f32; 4]; 4]) {
    use std::mem;
    let array = unsafe { mem::transmute::<&[[f32; 4]; 4], &[f32; 16]>(value) as &[f32] };
    js! {@(no_return)
      (@{self}).uniformMatrix4fv(@{location.deref()},false,@{&array})
    }
  }

  pub fn uniform_matrix_3fv(&self, location: &WebGLUniformLocation, value: &[[f32; 3]; 3]) {
    use std::mem;
    let array = unsafe { mem::transmute::<&[[f32; 3]; 3], &[f32; 9]>(value) as &[f32] };
    js! {@(no_return)
      (@{self}).uniformMatrix3fv(@{location.deref()},false,@{&array})
    }
  }

  pub fn uniform_matrix_2fv(&self, location: &WebGLUniformLocation, value: &[[f32; 2]; 2]) {
    use std::mem;
    let array = unsafe { mem::transmute::<&[[f32; 2]; 2], &[f32; 4]>(value) as &[f32] };
    js! {@(no_return)
      (@{self}).uniformMatrix2fv(@{location.deref()},false,@{&array})
    }
  }

  pub fn uniform_1i(&self, location: &WebGLUniformLocation, value: i32) {
    js! {@(no_return)
      (@{self}).uniform1i(@{location.deref()},@{value})
    }
  }

  pub fn uniform_1f(&self, location: &WebGLUniformLocation, value: f32) {
    js! {@(no_return)
      (@{self}).uniform1f(@{location.deref()},@{value})
    }
  }

  pub fn uniform_2f(&self, location: &WebGLUniformLocation, value: (f32, f32)) {
    js! {@(no_return)
      (@{self}).uniform2f(@{location.deref()},@{value.0},@{value.1})
    }
  }

  pub fn uniform_4f(&self, location: &WebGLUniformLocation, value: (f32, f32, f32, f32)) {
    js! { (@{self}).uniform4f(@{location.deref()},@{value.0},@{value.1},@{value.2},@{value.3}) }
  }

  pub fn create_vertex_array(&self) -> WebGLVertexArrayObject {
    let val = js! {
      return (@{&self}).createVertexArray()
    };
    WebGLVertexArrayObject(val.try_into().unwrap())
  }

  pub fn bind_vertex_array(&self, vao: &WebGLVertexArrayObject) {
    js! {@(no_return)
      (@{&self}).bindVertexArray(@{vao.deref()});
    }
  }

  pub fn unbind_vertex_array(&self) {
    js! {@(no_return)
      (@{&self}).bindVertexArray(0)
    }
  }

  pub fn cull_face(&self, mode: Culling) {
    js! {@(no_return)
      (@{self}).cullFace(@{mode as i32});
    }
  }

  pub fn get_program_parameter(&self, program: &WebGLProgram, pname: ShaderParameter) -> i32 {
    let res = js! { return (@{&self}).getProgramParameter(@{program.deref()},@{pname as u32}); };
    res.try_into().unwrap()
  }

  pub fn get_active_uniform(&self, program: &WebGLProgram, location: u32) -> WebGLActiveInfo {
    let res = js! { return @{self}.getActiveUniform(@{program.deref()},@{location}) };
    let name = js! { return @{&res}.name };
    let size = js! { return @{&res}.size };
    let kind = js! { return @{&res}.type };
    let k: u32 = kind.try_into().unwrap();
    use std::mem;
    WebGLActiveInfo::new(
      name.into_string().unwrap(),
      size.try_into().unwrap(),
      unsafe { mem::transmute::<u16, UniformType>(k as _) },
      res.try_into().unwrap(),
    )
  }

  pub fn get_active_attrib(&self, program: &WebGLProgram, location: u32) -> WebGLActiveInfo {
    let res = js! { return @{self}.getActiveAttrib(@{program.deref()},@{location}) };
    let name = js! { return @{&res}.name };
    let size = js! { return @{&res}.size };
    let kind = js! { return @{&res}.type };
    let k: u32 = kind.try_into().unwrap();
    use std::mem;
    WebGLActiveInfo::new(
      name.into_string().unwrap(),
      size.try_into().unwrap(),
      unsafe { mem::transmute::<u16, UniformType>(k as _) },
      res.try_into().unwrap(),
    )
  }

  pub fn tex_parameteri(&self, pname: TextureParameter, param: i32) {
    js! { return @{self}.texParameteri(@{TextureBindPoint::Texture2d as u32},@{pname as u32},@{param}) };
  }

  pub fn tex_parameterfv(&self, pname: TextureParameter, param: f32) {
    js! { return @{self}.texParameterf(@{TextureBindPoint::Texture2d as u32},@{pname as u32},@{param}) };
  }

  pub fn create_framebuffer(&self) -> WebGLFramebuffer {
    let val = js! {
      return @{self}.createFramebuffer()
    };
    WebGLFramebuffer(val.try_into().unwrap())
  }

  pub fn delete_framebuffer(&self, fb: &WebGLFramebuffer) {
    js! {@(no_return)
      return @{self}.deleteFramebuffer(@{fb.deref()})
    }
  }

  pub fn bind_framebuffer(&self, buffer: Buffers, fb: &WebGLFramebuffer) {
    js! {
      return @{self}.bindFramebuffer(@{fb.deref()})
    }
  }

  pub fn framebuffer_texture2d(
    &self,
    target: Buffers,
    attachment: Buffers,
    textarget: TextureBindPoint,
    texture: &WebGLTexture,
    level: i32,
  ) {
    js! {@(no_return)
      @{self}.framebufferTexture2D(@{target as u32},@{attachment as u32},@{textarget as u32},@{texture.deref()},@{level});
    }
  }

  pub fn unbind_framebuffer(&self, buffer: Buffers) {
    js! {@(no_return)
      @{self}.bindFramebuffer(@{buffer as u32},null);
    }
  }
}
