import odl
import numpy as np

def setup_forward_and_backward(resolution, samples):
    # Reconstruction space: discretized functions on the rectangle
    reco_space = odl.uniform_discr(
        min_pt=[-20, -20], max_pt=[20, 20], shape=[resolution, resolution], dtype='float32')

    # Angles: uniformly spaced, n = 1000, min = 0, max = pi
    angle_partition = odl.uniform_partition(0, np.pi, samples)

    # Detector: uniformly sampled, n = 500, min = -30, max = 30
    detector_partition = odl.uniform_partition(-30, 30, resolution)

    # Make a parallel beam geometry with flat detector
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection).
    radon = odl.tomo.RayTransform(reco_space, geometry)

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