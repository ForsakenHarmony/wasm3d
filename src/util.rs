use crate::engine::mesh::VertexPosTexNor;
use cgmath::Vector3;

fn create_box(dimensions: Option<Vector3<f32>>, position: Option<Vector3<f32>>) -> VertexPosTexNor {
  let Vector3 { x, y, z } = dimensions.unwrap_or(Vector3::new(1.0, 1.0, 1.0));
  let Vector3 { x: width, y: height, z: depth } = position.unwrap_or(Vector3::new(-x / 2.0, -y / 2.0, -z / 2.0));

  let fbl = Vector3::new(x        , y         , z + depth);
  let fbr = Vector3::new(x + width, y         , z + depth);
  let ftl = Vector3::new(x        , y + height, z + depth);
  let ftr = Vector3::new(x + width, y + height, z + depth);
  let bbl = Vector3::new(x        , y         , z);
  let bbr = Vector3::new(x + width, y         , z);
  let btl = Vector3::new(x        , y + height, z);
  let btr = Vector3::new(x + width, y + height, z);

  let positions = vec![
    //front
    fbl.x, fbl.y, fbl.z,
    fbr.x, fbr.y, fbr.z,
    ftl.x, ftl.y, ftl.z,
    ftl.x, ftl.y, ftl.z,
    fbr.x, fbr.y, fbr.z,
    ftr.x, ftr.y, ftr.z,

    //right
    fbr.x, fbr.y, fbr.z,
    bbr.x, bbr.y, bbr.z,
    ftr.x, ftr.y, ftr.z,
    ftr.x, ftr.y, ftr.z,
    bbr.x, bbr.y, bbr.z,
    btr.x, btr.y, btr.z,

    //back
    fbr.x, bbr.y, bbr.z,
    bbl.x, bbl.y, bbl.z,
    btr.x, btr.y, btr.z,
    btr.x, btr.y, btr.z,
    bbl.x, bbl.y, bbl.z,
    btl.x, btl.y, btl.z,

    //left
    bbl.x, bbl.y, bbl.z,
    fbl.x, fbl.y, fbl.z,
    btl.x, btl.y, btl.z,
    btl.x, btl.y, btl.z,
    fbl.x, fbl.y, fbl.z,
    ftl.x, ftl.y, ftl.z,

    //top
    ftl.x, ftl.y, ftl.z,
    ftr.x, ftr.y, ftr.z,
    btl.x, btl.y, btl.z,
    btl.x, btl.y, btl.z,
    ftr.x, ftr.y, ftr.z,
    btr.x, btr.y, btr.z,

    //bottom
    bbl.x, bbl.y, bbl.z,
    bbr.x, bbr.y, bbr.z,
    fbl.x, fbl.y, fbl.z,
    fbl.x, fbl.y, fbl.z,
    bbr.x, bbr.y, bbr.z,
    fbr.x, fbr.y, fbr.z
  ];

  let uvs = vec![
    //front
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    //right
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    //back
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    //left
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    //top
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    //bottom
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  ];

  let normals = vec![
    // front
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,

    // right
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,

    // back
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,

    // left
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,

    // top
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,

    // bottom
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0
  ];

  VertexPosTexNor {
    pos: positions,
    tex: uvs,
    nor: normals,
  }
}

