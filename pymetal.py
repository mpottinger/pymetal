import random
import re

import numba

import io
import logging
import os
import subprocess
from typing import List
import Foundation
import numpy
import objc   # Recommended: compile PyObjc from source to get latest version
import Metal as mtl


log = logging.getLogger(__name__)

class Metal():
    def __init__(self, source, func_name=None):
        self.pm = PyMetal()
        self.pm.opendevice()
        print("Metal shader initializing: ", source, "func_name: ", func_name)
        print('devices:', self.pm.lsdev())
        if source.find('.metallib') != -1:
            self.pm.openlibrary_compiled(source)
        elif source.find('.metal') != -1:
            self.pm.openlibrary(src=None, filename=source)
        elif source.find('kernel'):
            self.pm.openlibrary(source)

        if func_name is not None:
            self.fn = self.pm.getfn(func_name)
        else:  # search first func kernel void func_name
            self.fn = self.pm.getfn(re.search("kernel\s+void\s+(\w+)", source)[1])

        self.cqueue, self.cbuffer, self.buffer_list = None, None, None

    @staticmethod
    @numba.njit(cache=True, parallel=True)
    def rand(n, dtype):
        v = numpy.empty(n, dtype=dtype)
        for i in numba.prange(n):
            v[i] = random.uniform(0, 1)
        return v

    # replace search_str for rpl_str in input file file_in generating file_out
    @staticmethod
    def file_replace(file_in, file_out, search_str, rpl_str):
        with open(file_in, 'r') as file: file_data = file.read()
        kernel_func = re.search("kernel\s+void\s+(\w+)", file_data)[1]
        with open(file_out, 'w') as file: file.write(file_data.replace(search_str, rpl_str))
        return file_out, kernel_func

    def buffer(self, data):
        if type(data).__module__ == numpy.__name__:
            return self.pm.numpybuffer(data)
        if type(data) is int:
            return self.pm.intbuffer(data)
        if type(data) is float:
            return self.pm.floatbuffer(data)

    def float_buf(self, data):
        return self.buffer(numpy.array(data, dtype=numpy.float32))

    def int_buf(self, data):
        return self.buffer(numpy.array(data, dtype=numpy.int32))

    def empty(self, size):
        return self.pm.emptybuffer(size)

    def empty_int(self, size):
        return self.empty(size * numpy.dtype(numpy.int32).itemsize)

    def empty_float(self, size):
        return self.empty(size * numpy.dtype(numpy.float32).itemsize)

    def set_buffers(self, buffers=None, threads=None):  # set of buffers
        if self.cqueue is None:
            print("initializing metal command queue")
            self.cqueue = self.pm.getqueue()
        self.cbuffer = self.pm.getCommandBuffer(self.cqueue)
        self.buffer_list = buffers
        self.pm.enqueue_compute(cbuffer=self.cbuffer, func=self.fn, threads=threads, buffers=buffers)

    def run(self):
        self.pm.start_process(self.cbuffer)
        self.pm.wait_process(self.cbuffer)

    def get_buffer(self, buf, dtype):
        return self.pm.buf2numpy(buf, dtype)

    # methods to copy data to existing buffers
    def copy_to_buffer(self, buf, data):
        self.pm.copynumpy2buf(buf, data)





class PyMetal():
    PixelFormatRGBA8UNorm = mtl.MTLPixelFormatRGBA8Unorm
    StorageModeManaged = mtl.MTLResourceStorageModeManaged

    def __init__(self):
        self.dev = None
    def setopt(self, vv, opts: dict):
        for k, v in opts.items():
            fn = "set" + k + "_"
            if hasattr(vv, fn):
                log.debug("set(%s) %s <- %s", type(vv).__name__, k, v)
                getattr(vv, fn)(v)
            else:
                log.error("cannot set(%s) <- %s (%s not found)",
                          type(vv).__name__, k, v, fn)
        return vv

    def maxvalues(self) -> dict:
        res = {}
        for k in filter(lambda f: f.startswith("max") and not f.endswith("_"), dir(self.dev)):
            name = k[3:]
            res[name] = getattr(self.dev, k)()
        return res

    def configs(self) -> dict:
        res = {}
        for k in filter(lambda f: f.endswith("Config"), dir(self.dev)):
            name = k[:-6]
            res[name] = getattr(self.dev, k)()
        return res

    def logmethods(self, obj, pat=None):
        if pat is None:
            res = dir(obj)
        else:
            res = list(filter(lambda f: f.find(pat) != -1, dir(obj)))
        log.debug("%s %s %s", type(obj), pat, res)
        return res

    def lsdev(self):
        return mtl.MTLCopyAllDevices()

    def device2str(self, d) -> List[str]:
        def yes(x):
            if x:
                return "yes"
            return "no"

        def supported(x):
            if x:
                return "✅ supported"
            return "❌ not supported"

        res = []
        res.append(d.name() + ":")
        res.append("	• low-power: " + yes(d.isLowPower()))
        res.append("	• removable: " + yes(d.isRemovable()))
        res.append("	• configured as headless: " + yes(d.isHeadless()))
        res.append("	• registry ID: " + str(d.registryID()))
        res.append("")
        res.append("	Feature Sets:")
        for k in filter(lambda f: f.startswith("MTLFeatureSet_"), dir(mtl)):
            name = k[14:]
            val = getattr(mtl, k)
            res.append("	• %s: %s" %
                       (name, supported(d.supportsFeatureSet_(val))))
        return res

    def opendevice(self, name=None):
        if name is None:
            self.dev = mtl.MTLCreateSystemDefaultDevice()
        else:
            devs = list(filter(lambda f: f.name() == name, self.lsdev()))
            if len(devs) == 1:
                self.dev = devs[0]
            else:
                raise Exception("no such device: %s / %s" %
                                (name, list(map(lambda f: f.name(), self.lsdev()))))

    def openlibrary(self, src=None, filename=None, **kwargs):
        if filename is not None:
            if isinstance(filename, str):
                src = open(filename).read()
            elif isinstance(filename, io.IOBase):
                src = filename.read()
            elif isinstance(filename, (list, tuple)):
                src = "\n".join(map(lambda f: open(f).read(), filename))
        opts = mtl.MTLCompileOptions.new()
        self.setopt(opts, kwargs)
        # err = Foundation.NSError.alloc()
        log.debug("openlibrary(source)")
        self.lib = self.dev.newLibraryWithSource_options_error_(
            src, opts, objc.NULL)[0]
        if self.lib is None:
            log.error("compile error?: %s", src)

    def openlibrary_compiled(self, filename, **kwargs):
        log.debug("openlibrary(compiled): %s", filename)
        self.lib = self.dev.newLibraryWithFile_error_(filename, objc.NULL)
        if self.lib is None:
            log.error("load error?: %s", filename)

    def openlibrary_default(self, **kwargs):
        log.debug("openlibrary(default)")
        self.lib = self.dev.newDefaultLibrary()
        if self.lib is None:
            log.error("load error?(default)")

    def getfn(self, name):
        return self.lib.newFunctionWithName_(name)

    def emptybuffer(self, size, label=None, opts=0):
        res = self.dev.newBufferWithLength_options_(size, opts)
        if label is not None:
            res.setLabel_(label)
        return res

    def numpybuffer(self, data, label=None, opts=0):
        buf = self.emptybuffer(data.nbytes, label, opts)
        buf.contents().as_buffer(buf.length())[:] = data.tobytes()
        return buf

    def bytesbuffer(self, data, label=None, opts=0):
        res = self.dev.newBufferWithLength_options_(len(data), opts)
        if label is not None:
            res.setLabel_(label)
        ibc = res.contents().as_buffer(res.length())
        ibc[:] = data
        return res

    def intbuffer(self, i):
        return self.numpybuffer(numpy.array(i, dtype=numpy.int32))

    def uintbuffer(self, i):
        return self.numpybuffer(numpy.array(i, dtype=numpy.uint32))

    def floatbuffer(self, f):
        return self.numpybuffer(numpy.array(f, dtype=numpy.float32))

    def syncbuffer(self, buffer, start=0, length=None):
        # sync from CPU to GPU
        if length is None:
            length = buffer.length() - start
        buffer.didModifyRange_(Foundation.NSRange(
            location=start, length=length))

    def emptytexture(self, size, label=None, opts=0):
        desc = mtl.MTLTextureDescriptor.new()
        log.info("texture desc: %s", desc)
        # TODO

    def buf2byte(self, buf):
        return buf.contents().as_buffer(buf.length())

    def buf2numpy(self, buf, dtype):
        return numpy.frombuffer(buf.contents().as_buffer(buf.length()), dtype=dtype)

    # methods to copy numpy data to existing buffers
    def copynumpy2buf(self,buf,data):
        # make sure data is a numpy array
        if not isinstance(data, numpy.ndarray):
            raise Exception("src data is not a numpy array: %s" % type(data))
        # make sure data is the right size
        if data.nbytes != buf.length():
            raise Exception("src data is wrong size: %s != %s" % (data.nbytes, buf.length()))
        self.buf2byte(buf)[:] = data.tobytes()


    def getqueue(self, **kwargs):
        cqueue = self.dev.newCommandQueue()
        self.setopt(cqueue, kwargs)
        return cqueue

    def getCommandBuffer(self, cqueue, **kwargs):
        cbuffer = cqueue.commandBuffer()
        self.setopt(cbuffer, kwargs)
        return cbuffer


    def getmtlsize(self, arg):
        if isinstance(arg, int):
            return mtl.MTLSize(width=arg, height=1, depth=1)
        return mtl.MTLSize(**arg)

    def enqueue_compute(self, cbuffer, func, buffers, threads=None, iters=None, label=None):
        desc = mtl.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()
        if iters is not None:
            bufmax = iters
        # threads
        if threads is None:
            # number of thread per group
            w = state.threadExecutionWidth()
            h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
            log.debug("w,h=%d,%d, bufmax=%d", w, h, bufmax)
            tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})
            # number of thread group per grid
            # w2 = max(1, int((bufmax + w * h - 1) / (w * h)))
            w2 = int(max(1, (bufmax + w - 1) / w))
            ntg = self.getmtlsize(w2)
            log.debug("threads: ntg=%s, tpg=%s", ntg, tpg)
            # encoder.dispatchThreadgroups_threadsPerThreadgroup_(ntg, tpg)
            encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        else:
            assert len(threads) >= 2
            # number of thread group
            ntg = mtl.MTLSize(width=threads[0], height=threads[1], depth=1)
            # number of thread per group
            _w = state.threadExecutionWidth()
            _h = state.maxTotalThreadsPerThreadgroup() / _w
            tpg = mtl.MTLSize(width=_w, height=_h, depth=1)
            log.debug("threads: %s %s", ntg, tpg)
            encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        log.debug("encode(compute) %s", label)
        encoder.endEncoding()

    def runThread2d(self, w, h):
        '''
        MTLSize threadsPerGrid = MTLSizeMake(width, height, 1);
  NSUInteger _w = pipeline.threadExecutionWidth;
  NSUInteger _h = pipeline.maxTotalThreadsPerThreadgroup / _w;
  MTLSize threadsPerThreadgroup = MTLSizeMake(_w, _h, 1);
  [commandEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        '''
        # self.dev.runThreadsWidth(w, h)
        threadsPerGrid = self.getmtlsize(w, h, 1)
        pipeline = mtl.MTLComputePipelineDescriptor.new()
        _w = pipeline.threadExecutionWidth
        _h = pipeline.maxTotalThreadsPerThreadgroup / _w
        threadsPerThreadgroup = self.getmtlsize(_w, _h, 1)

    def runThread(self, cbuffer, func, buffers, threads=None, label=None):
        desc = mtl.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()

        # threads

        # number of thread per group
        w = state.threadExecutionWidth()
        h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
        log.debug("w,h=%d,%d, bufmax=%d", w, h, bufmax)
        tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})

        # number of thread per grig
        ntg = self.getmtlsize(threads)
        log.debug("threads: %s %s", ntg, tpg)

        encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        log.debug("encode(compute) %s", label)
        encoder.endEncoding()

    def enqueue_blit(self, cbuffer, texture=None, label=None):
        encoder = cbuffer.blitCommandEncoder()
        if label is not None:
            encoder.setLabel_(label)
        if texture is not None:
            encoder.synchronizeResource_(texture)
        log.debug("encode(blit) %s", label)
        encoder.endEncoding()

    def enqueue_render(self, cbuffer, buffers):
        # TBD
        log.error("enqueue_render: not implemented")
        pass

    def start_process(self, cbuffer):
        log.debug("start compute")
        cbuffer.commit()

    def wait_process(self, cbuffer):
        log.debug("wait")
        cbuffer.waitUntilCompleted()
        log.debug("finished")

    def compile(self, source, outfn):
        basepath = "/Applications/Xcode.app/Contents/Developer/Toolchains"
        metalpath = os.path.join(basepath, "XcodeDefault.xctoolchain",
                                 "usr", "metal", "macos", "bin", "metal")
        cmd = [metalpath, "-x", "metal", "-", "-o", outfn]
        log.debug("compile: %s", cmd)
        p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p1.communicate(source.encode("utf-8"))
