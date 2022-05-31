import odl
import numpy as np

def setup_forward_and_backward(resolution, samples):
    # Reconstruction space: discretized functions on the rectangle
    reco_space = odl.uniform_discr(
        min_pt=[-resolution//2+1, -resolution//2+1], max_pt=[resolution//2, resolution//2], shape=[resolution, resolution], dtype='float32')

    # Angles: uniformly spaced, min = 0, max = pi
    #angle_partition = odl.uniform_partition(0, 2 * np.pi, samples)

    # Make a parallel beam geometry
    geometry = odl.tomo.parallel_beam_geometry(reco_space, samples, det_shape=resolution)

    # Ray transform (= forward projection).
    radon = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # Fourier transform in detector direction
    fourier = odl.trafos.FourierTransform(radon.range, axes=[1])
    # Create ramp in the detector direction
    ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
    # Create ramp filter via the convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * fourier

    # Create filtered back-projection by composing the back-projection (adjoint)
    # with the ramp filter.
    fbp = radon.adjoint * ramp_filter

    return radon, fbp