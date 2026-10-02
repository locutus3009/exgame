// SPDX-License-Identifier: MIT

use super::{AxialSpringDamper, AxialSpringDamperType, AxialSpringForce};
use crate::{Joint, JointFromParams};
use aristotle::World;
use bytemuck::Pod;
use peano::prelude::*;
use std::sync::Arc;

#[derive(Debug)]
pub struct SimpleSpringDamper;

impl<T> JointFromParams<T> for SimpleSpringDamper
where
    T: Scalar + StandardPart + Pod,
{
    fn shader_name(&self) -> &'static str {
        "SimpleSpringDamper"
    }
    fn n_params(&self) -> usize {
        10
    }
    fn build_from_params(&self, world: Arc<World>, params: &[T]) -> Joint<T> {
        AxialSpringDamper::<T>::builder(world.clone(), Self)
            .a(Vector3::from([params[0], params[1], params[2]]))
            .b(Vector3::from([params[3], params[4], params[5]]))
            .rest(params[6])
            .stiffness(params[7])
            .damping(params[8])
            .softening(params[9])
            .build()
    }
}

impl<T: Scalar> AxialSpringForce<T> for SimpleSpringDamper {
    fn into_enum(self) -> AxialSpringDamperType {
        AxialSpringDamperType::SimpleSpringDamper(self)
    }
}
