extern crate application;
extern crate webgl;
extern crate stdweb;

use application::*;
use webgl::*;

use stdweb::web::INode;

use std::mem::size_of;

mod app;

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

impl<T> IntoBytes for Vec<T> {
    fn into_bytes(self) -> Vec<u8> {
        let len = size_of::<T>() * self.len();
        unsafe {
            let slice = self.into_boxed_slice();

            let out = Vec::<u8>::from_raw_parts(Box::into_raw(slice) as _, len, len);
            out
        }
    }
}

pub fn main() -> Result<(), Box<std::error::Error>> {
    let size = (800, 600);
    let config = AppConfig::new("Test", size);
    let mut app = App::new(config);

    let vertices: Vec<f32> = vec![-0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let indices: Vec<u16> = vec![0, 1, 2];
    let count = indices.len();

    let elem = stdweb::web::document().create_element("canvas");
    stdweb::web::document().append_child(&elem);

    let gl = WebGLRenderingContext::new(&elem);

    // Create an empty buffer object to store vertex buffer
    let vertex_buffer = gl.create_buffer();

    // Bind appropriate array buffer to it
    gl.bind_buffer(BufferKind::Array, &vertex_buffer);

    // Pass the vertex data to the buffer
    gl.buffer_data(BufferKind::Array, &vertices.into_bytes(), DrawMode::Static);

    // Unbind the buffer
    gl.unbind_buffer(BufferKind::Array);

    // Create an empty buffer object to store Index buffer
    let index_buffer = gl.create_buffer();

    // Bind appropriate array buffer to it
    gl.bind_buffer(BufferKind::ElementArray, &index_buffer);

    // Pass the vertex data to the buffer
    gl.buffer_data(
        BufferKind::ElementArray,
        &indices.into_bytes(),
        DrawMode::Static,
    );

    // Unbind the buffer
    gl.unbind_buffer(BufferKind::ElementArray);

    /*================ Shaders ====================*/

    // Vertex shader source code
    let vert_code = "attribute vec3 coordinates;
    void main(void) {
        gl_Position = vec4(coordinates, 1.0);
    }";

    // Create a vertex shader object
    let vert_shader = gl.create_shader(ShaderKind::Vertex);

    // Attach vertex shader source code
    gl.shader_source(&vert_shader, vert_code);

    // Compile the vertex shader
    gl.compile_shader(&vert_shader);

    //fragment shader source code
    let frag_code = "void main(void) {
        gl_FragColor = vec4(1, 0.5, 0.0, 1);
    }";
    // Create fragment shader object
    let frag_shader = gl.create_shader(ShaderKind::Fragment);

    // Attach fragment shader source code
    gl.shader_source(&frag_shader, frag_code);

    // Compile the fragmentt shader
    gl.compile_shader(&frag_shader);

    // Create a shader program object to store
    // the combined shader program
    let shader_program = gl.create_program();

    // Attach a vertex shader
    gl.attach_shader(&shader_program, &vert_shader);

    // Attach a fragment shader
    gl.attach_shader(&shader_program, &frag_shader);

    // Link both the programs
    gl.link_program(&shader_program);

    // Use the combined shader program object
    gl.use_program(&shader_program);

    /*======= Associating shaders to buffer objects =======*/

    // Bind vertex buffer object
    gl.bind_buffer(BufferKind::Array, &vertex_buffer);

    // Bind index buffer object
    gl.bind_buffer(BufferKind::ElementArray, &index_buffer);

    // Get the attribute location
    let coord = gl.get_attrib_location(&shader_program, "coordinates".into())
        .unwrap();

    // Point an attribute to the currently bound VBO
    gl.vertex_attrib_pointer(coord, AttributeSize::Three , DataType::Float, false, 0, 0);

    // Enable the attribute
    gl.enable_vertex_attrib_array(coord);

    /*=========Drawing the triangle===========*/

    // Clear the canvas
    gl.clear_color(0.5, 0.5, 0.5, 0.9);

    // Enable the depth test
    gl.enable(Flag::DepthTest);

    // Clear the color buffer bit
    gl.clear(BufferBit::Color);
    gl.clear(BufferBit::Depth);

    // Set the view port
    gl.viewport(0, 0, size.0, size.1);

    app.run(move |_t:&mut App| {
        gl.clear(BufferBit::Color);
        gl.clear(BufferBit::Depth);
        gl.clear_color(1.0, 1.0, 1.0, 1.0);
        gl.draw_elements(Primitives::Triangles, count, DataType::U16, 0);
    });

    Ok(())
}