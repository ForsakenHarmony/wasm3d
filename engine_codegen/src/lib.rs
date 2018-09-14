#![recursion_limit = "1000"]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro2;
extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use std::collections::HashMap;

use proc_macro::TokenStream;
use proc_macro2::Span;

use syn::{DeriveInput, Data, Fields, spanned::Spanned, Attribute, Meta, NestedMeta, Lit, Ident, Type, TypePath};

fn get_attr_map(attr: &Attribute) -> Option<(String, HashMap<String, Lit>, Span)> {
  let meta = attr.interpret_meta();

  let meta_list = match meta {
    Some(Meta::List(ref meta_list)) => meta_list,
    _ => return None,
  };

  let ident = meta_list.ident.to_string();

  let mut attr_map = HashMap::new();

  for meta in meta_list.nested.iter() {
    let value = match meta {
      NestedMeta::Meta(Meta::NameValue(ref value)) => value,
      _ => continue,
    };

    let name = value.ident.to_string();

    attr_map.insert(name, value.lit.clone());
  }

  Some((ident, attr_map, meta.span()))
}

fn get_attrs_map(attrs: &Vec<Attribute>, name: &str) -> Option<(HashMap<String, Lit>, Span)> {
  attrs.iter().filter_map(get_attr_map).find(|(attr_name, _, _)| attr_name == name).map(|(_, map, span)| (map, span))
}

fn emit_error(span: proc_macro2::Span, error: &str) -> TokenStream {
  span.unstable()
      .error(error)
      .emit();
  TokenStream::new()
}

#[proc_macro_derive(VertexFormat, attributes(vertex))]
pub fn derive_vertex_format(input: TokenStream) -> TokenStream {
  let input: DeriveInput = parse_macro_input!(input as DeriveInput);

  let name = input.ident;

  let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

  let struct_data = match input.data {
    Data::Struct(struct_data) => struct_data,
    Data::Enum(enum_data) => return emit_error(enum_data.enum_token.span, "Only structs with named fields are supported"),
    Data::Union(union_data) => return emit_error(union_data.union_token.span, "Only structs with named fields are supported")
  };

  let fields = match struct_data.fields {
    Fields::Named(fields) => fields,
    _ => return emit_error(struct_data.fields.span(), "Only structs with named fields are supported")
  };

  let attributes = match get_attrs_map(&input.attrs, "vertex") {
    Some(map) => map,
    None => return emit_error(struct_data.struct_token.span, "Missing vertex attribute with `flags`")
  };

  let flags = match attributes.0.get("flags") {
    Some(flags_lit) => match flags_lit {
      Lit::Str(flags) => {
        Ident::new(&flags.value(), flags_lit.span())
//        let flags = flags.value();
//        quote_spanned!(flags_lit.span() => #flags)
//        let mut _s = ::__rt::TokenStream::new();
//        let _span = (flags_lit.span());
//        ::ToTokens::to_tokens(&flags, &mut _s);
//        _s
      },
      _ => return emit_error(flags_lit.span(), "`name` has to be a string")
    },
    None => return emit_error(attributes.1, "Missing `flags` in attribute")
  };

  let mut field_things = Vec::new();

  for field in fields.named {
    let name = field.ident.expect("We only have named structs anyways");

    let attributes = match get_attrs_map(&field.attrs, "vertex") {
      Some(map) => map,
      None => return emit_error(name.span(), "Missing vertex attribute with `name` and `size`")
    };

    let loc = match attributes.0.get("loc") {
      Some(name_lit) => match name_lit {
        Lit::Int(name) => {
          name.clone()
        },
        _ => return emit_error(name_lit.span(), "`name` has to be an int")
      },
      None => return emit_error(attributes.1, "Missing `name` in attribute")
    };

    let size = match attributes.0.get("size") {
      Some(size_lit) => match size_lit {
        Lit::Int(size) => {
          match size.value() {
            1..=4 => size.clone(),
            _ => return emit_error(size_lit.span(), "`size` has to be an int in the range 1..=4")
          }
        }
        _ => return emit_error(size_lit.span(), "`size` has to be an int in the range 1..=4")
      },
      None => return emit_error(attributes.1, "Missing `size` in attribute")
    };

    let tokens_f32 = quote!(Vec<f32>).into();
    let tokens_u8 = quote!(Vec<u8>).into();

    let mut parsed_f32 = parse_macro_input!(tokens_f32 as TypePath);
    let mut parsed_u8 = parse_macro_input!(tokens_u8 as TypePath);

    let data_type = match field.ty {
      Type::Path(ref path) => {
//        match path {
//          _ if path == &parsed_f32 => quote!(DataType::Float),
//          _ if path == &parsed_f32 => quote!(DataType::Float),
//          _ => return emit_error(field.ty.span(), "Currently only Vec<f32> and Vec<u8> are supported"),
//        }
        if path == &parsed_f32 {
          quote!(DataType::Float)
        } else if path == &parsed_u8 {
          quote!(DataType::U8)
        } else {
          return emit_error(field.ty.span(), "Currently only Vec<f32> and Vec<u8> are supported")
        }
      },
      _ => return emit_error(field.ty.span(), "Currently only Vec<f32> and Vec<u8> are supported")
    };

    let buffer_name = Ident::new(&(name.to_string() + "_buffer"), name.span());

    let buffer = quote!(let #buffer_name = program.create_vertex_buffer(#data_type, #size, self.#name.as_slice()););
    let buffer_attr = quote!(.vertex_attribute_buffer(#loc, &#buffer_name));

    field_things.push((name, size, buffer_name, buffer, buffer_attr))
  }

  let buffers = field_things.iter().map(|(_, _, _, buffer, _)| buffer).collect::<Vec<_>>();
  let buffer_attrs = field_things.iter().map(|(_, _, _, _, buffer_attr)| buffer_attr).collect::<Vec<_>>();
  let buffer_names = field_things.iter().map(|(_, _, buffer_name, _, _)| buffer_name).collect::<Vec<_>>();

  let (ref first_name, ref first_size, _, _, _) = field_things[0];

  let expanded = quote! {
    impl #impl_generics ::engine::mesh::VertexFormat for #name #ty_generics #where_clause {
      fn flags() -> VertexFlags {
        VertexFlags::#flags
      }

      fn create_buffers(&self, program: &ShaderProgram, indices: &[u16]) -> (VAO, Vec<VBO>, VBO) {
        #(#buffers)*

        let index_buffer = program.create_index_buffer(DataType::U16, 3, indices);

        let mut vao = program.create_vertex_array();
        vao
          #(#buffer_attrs)*
          .index_buffer(&index_buffer);

        (vao, vec![#(#buffer_names),*], index_buffer)
      }

      fn vertex_count(&self) -> usize {
        self.#first_name.len() / #first_size
      }
    }
  };

//  println!("{}", expanded);

  TokenStream::from(expanded)
}
