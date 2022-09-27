'''
test for DomainColoring metal app. with several funcs directly replace in source code
'''
from timeit import default_timer as lap
import cv2
import numpy as np

from pymetal import Metal

predef_funcs = ['acos(c(1,2)*log(sin(z*z*z-1)/z))', 'c(1,1)*log(sin(z*z*z-1)/z)', 'c(1,1)*sin(z)',
                'z + z*z/sin(z*z*z*z-1)', 'log(sin(z))', 'cos(z)/(sin(z*z*z*z-1))', 'z*z*z*z*z*z-1',
                '(z*z-1) * pow((z-c(2,1)),2) / (z*z+c(2,1))', 'sin(z)*c(1,2)', 'sin(c(1)/z)', 'sin(z)*sin(c(1)/z)',
                'c(1)/sin(c(1)/sin(z))', 'z', '(z*z+1)/(z*z-1)', '(z*z+c(1))/z', '(z+3)*pow((z+1),2)',
                'pow((z/c(2)),2)*(z+c(1,2))*(z+c(2,2))/(z*z*z)', '(z*z)-0.75-c(0,0.2)',
                'z*sin(z/cos(sin(c(2.2)/z)*tan(z)))']

# replace in original metal file %%FUNC%% by string z expression

while True:
    for fz in predef_funcs:
        metal_file, func_name = Metal.file_replace(file_in='dc.metal', file_out='dcz.metal',
                                                   search_str='%%FUNC%%', rpl_str=fz)
        t0 = lap()
        m = Metal(metal_file, func_name)
        t0 = lap() - t0

        sz = 4
        w, h = 640 * sz, 480 * sz

        print(f'compiled in {t0:.2} run dc on size {w, h}', end='')

        bgeo = m.buffer(np.array([w, h], dtype=np.int32))
        bpix = m.empty(w * h * 4)

        m.set_buffers(buffers=(bpix, bgeo), threads=(w, h))

        t = lap()

        m.run()

        pix = m.get_buffer(bpix, dtype=np.int32)
        print(f' done in {lap() - t:.3}", processing image...')

        # convert raw RGBA int32 buffer to openCV image
        pix = pix.view(dtype=np.uint8).reshape((h, w, 4))
        pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2BGRA)
        pix = cv2.cvtColor(pix, cv2.COLOR_BGRA2BGR)

        # show image
        cv2.imshow('dc', pix)
        cv2.waitKey(1)

pass
