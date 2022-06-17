import odl
import numpy as np
import torch

def setup_forward_and_backward(resolution, samples, angles=None):
    reco_space = odl.uniform_discr(
        min_pt=[-resolution//2+1, -resolution//2+1], max_pt=[resolution//2, resolution//2], shape=[resolution, resolution], dtype='float32',impl='numpy')

    # Angles: uniformly spaced, n = 1000, min = 0, max = pi
    if angles is None:
        angle_partition = odl.uniform_partition(0, 2 * np.pi, samples)
    else:
        angle_partition = odl.nonuniform_partition(angles, min_pt = 0, max_pt =  2 * np.pi)

    # Detector: uniformly sampled, n = 500, min = -30, max = 30
    detector_partition = odl.uniform_partition(-resolution+1, resolution, resolution*2)

    # Make a parallel beam geometry with flat detector
    geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)
    # geometry = odl.tomo.parallel_beam_geometry(reco_space, samples, det_shape=resolution*2)

    # Ray transform (= forward projection).
    radon = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # s = odl.tomo.backends.astra_cuda.astra_cuda_bp_scaling_factor(radon.range,reco_space,geometry)
    # Fourier transform in detector direction
    fourier = odl.trafos.FourierTransform(radon.range, axes=[1],impl='numpy')
    # Create ramp in the detector 'direction'
    ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
    # Create ramp filter via the convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * (fourier)

    # Create filtered back-projection by composing the back-projection (adjoint)
    # with the ramp filter.
    fbp = radon.adjoint * ramp_filter

    return radon, fbp, torch.nn.ConstantPad2d((resolution//2, resolution//2, 0, 0), 0.0 )