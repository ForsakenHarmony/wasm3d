#![recursion_limit="1000"]

extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro::TokenStream;

use syn::{DeriveInput};

#[proc_macro_derive(VertexFormat)]
pub fn derive_vertex_format(input: TokenStream) -> TokenStream {
  println!("test");

  let input = parse_macro_input!(input as DeriveInput);

//  println!("test");
//  println!("{:#?}", input);

  let name = input.ident;

  let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

  let expanded = quote! {
    impl #impl_generics ::engine::mesh::VertexFormat for #name #ty_generics #where_clause {
      type Buffers = (VBO, VBO);
      const FLAGS: VertexFlags = VertexFlags::PosTex;

      fn create_buffers(&self, program: &ShaderProgram) -> Self::Buffers {
        let pos_buffer = program.create_buffer(AttributeSize::Three, "a_position", DataType::Float);
        let tex_buffer = program.create_buffer(AttributeSize::Two, "a_texcoord", DataType::Float);

        pos_buffer.set_data_f32(self.pos.as_slice());
        tex_buffer.set_data_f32(self.tex.as_slice());

        (pos_buffer, tex_buffer)
      }

      fn vertex_count(&self) -> usize {
        self.pos.len() / 3
      }
    }
  };

  TokenStream::from(expanded)
}
