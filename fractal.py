'''
test for fractal metal app. with several funcs directly replace in source code
'''
from timeit import default_timer as lap

import numpy as np
import cv2

from pymetal import Metal

t0 = lap()
m = Metal('fractal.metal', 'fractal')
t0 = lap() - t0

sz = 4
w, h = 640 * sz, 480 * sz

print(f'compiled in {t0:.2} run fractal on size {w, h}={w * h} pix', end='')

# create i/o buffers to match kernel func. params
c = np.array([0.5, 0], dtype=np.float32)
r = np.array([-2, 2], dtype=np.float32)
bsize = m.int_buf([w, h]) # input: size(w,h), center(x,y), range(x,y)
bcenter = m.float_buf(c)
brange = m.float_buf(r)

bpix = m.empty_int(w * h) # output


while True:
    # random center and range
    c = np.random.uniform(-1, 1, 2).astype(np.float32)
    r = np.random.uniform(-2, 2, 2).astype(np.float32)

    print(f' center={c}, range={r}', end='')
    m.copy_to_buffer(bcenter, c)
    m.copy_to_buffer(brange, r)
    m.set_buffers(buffers=(bpix, bsize, bcenter, brange), threads=(w, h))

    t = lap()

    m.run()

    pix = m.get_buffer(bpix, dtype=np.int32)
    print(f' done in {lap() - t:.3}", processing image...')

    # convert raw RGBA int32 buffer to openCV image
    pix = pix.view(dtype=np.uint8).reshape((h, w, 4))
    pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2BGRA)
    pix = cv2.cvtColor(pix, cv2.COLOR_BGRA2BGR)

    cv2.imshow('fractal', pix)
    cv2.waitKey(1)
