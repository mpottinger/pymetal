
# Metal Mesh Rendering in Python
# -----------------------------
# Here is an example of how to draw a basic triangle in Apple's Metal Graphics API using PyObjC in Python on MacOS.
# This code will create a Metal device, command queue, texture, render pipeline,
# command buffer, and render pass descriptor,
# vertex buffer, projection and view matrices, etc., and finally draw a triangle using Metal.
# Then, it will copy the contents of the Metal texture to an OpenCV image, and display it using OpenCV.

# OpenCV imshow is not the fastest way to display 3D rendered images, but it is the easiest way to display images in Python,
# and is flexible.

# The goal here is to make the power of Metal 3D rendering available to create a fast and flexible render to texture/image
# based 3D rendering that allows the rendered images to be usable in Python,
# for further processing via Metal compute shaders, or use in machine learning and computer vision applications.

import ctypes
import math
import time
from random import random

# Import the necessary modules and classes
import objc
import Metal
import Quartz
import Foundation
import cv2
import numpy as np

# Get the resolution of the screen
screen = Quartz.CGMainDisplayID()
width = Quartz.CGDisplayPixelsWide(screen)
height = Quartz.CGDisplayPixelsHigh(screen)

print("Screen resolution: ", width, "x", height)
aspect = width / height

WIDTH = 640
HEIGHT = 480

# make WIDTH / HEIGHT the same aspect ratio as the screen
#WIDTH = int(HEIGHT * aspect)


if __name__ == '__main__':
    # Create a Metal device
    device = Metal.MTLCopyAllDevices()[0]

    #print(dir(device))
    # Create a Metal command queue
    commandQueue = device.newCommandQueue()

    # Create a Metal texture to use for offscreen rendering (color)
    textureDescriptorColor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatBGRA8Unorm, WIDTH, HEIGHT , False)
    # Create the texture using the texture descriptor
    color_texture = device.newTextureWithDescriptor_(textureDescriptorColor)

    # Create a Metal texture to use for offscreen rendering (depth)
    textureDescriptorDepth = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatDepth32Float, WIDTH, HEIGHT , False)
    # Create the texture using the texture descriptor
    depth_texture = device.newTextureWithDescriptor_(textureDescriptorDepth)

    # Create a Metal render pipeline
    # Load the source code for the vertex shader and fragment shader from mesh_shader.metal
    # Create the shader library from the source code
    shader_source = open('mesh_shader.metal', 'r').read()

    # Compile the shader library with default options
    library, result = device.newLibraryWithSource_options_error_(shader_source, None, None)
    print("Shader library compilation result: %s" % result)
    vertexFunction = library.newFunctionWithName_("vertex_main")
    fragmentFunction = library.newFunctionWithName_("fragment_main")

    # Create a vertex descriptor
    vertexDescriptor = Metal.MTLVertexDescriptor.new()
    # Specify the layout of the vertex buffer
    vertexDescriptor.attributes().objectAtIndexedSubscript_(0).setFormat_(Metal.MTLVertexFormatFloat4)
    vertexDescriptor.attributes().objectAtIndexedSubscript_(0).setOffset_(0)
    vertexDescriptor.attributes().objectAtIndexedSubscript_(0).setBufferIndex_(0)
    vertexDescriptor.layouts().objectAtIndexedSubscript_(0).setStride_(16)

    vertexDescriptor.attributes().objectAtIndexedSubscript_(1).setFormat_(Metal.MTLVertexFormatFloat4)
    vertexDescriptor.attributes().objectAtIndexedSubscript_(1).setOffset_(16)
    vertexDescriptor.attributes().objectAtIndexedSubscript_(1).setBufferIndex_(0)
    vertexDescriptor.layouts().objectAtIndexedSubscript_(0).setStride_(32)
    print("vertex descriptor created")

    renderPipelineDescriptor = Metal.MTLRenderPipelineDescriptor.new()
    renderPipelineDescriptor.setVertexFunction_(vertexFunction)
    renderPipelineDescriptor.setFragmentFunction_(fragmentFunction)
    renderPipelineDescriptor.setVertexDescriptor_(vertexDescriptor)
    bgra8Unorm = 80
    renderPipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(bgra8Unorm)
    renderPipelineDescriptor.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
    print(renderPipelineDescriptor)
    print("render pipeline descriptor created")

    renderPipeline, error = device.newRenderPipelineStateWithDescriptor_error_(renderPipelineDescriptor, None)
    print(renderPipeline)
    if error:
        print(error)
        quit()
    print("render pipeline created")

    # Create a Metal render pass descriptor
    renderPassDescriptor = Metal.MTLRenderPassDescriptor.new()
    # color attachment
    renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setTexture_(color_texture)
    renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setLoadAction_(Metal.MTLLoadActionClear)
    # store action is set to store so that the texture can be read from later
    renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setStoreAction_(Metal.MTLStoreActionStore)
    renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))
    # depth attachment
    # Set the depth texture as the depth attachment of the render pass descriptor
    # Create an instance of the MTLRenderPassDepthAttachmentDescriptor class
    depthAttachmentDescriptor = Metal.MTLRenderPassDepthAttachmentDescriptor.new()
    # Set the texture property of the depth attachment descriptor to the depth texture
    depthAttachmentDescriptor.setTexture_(depth_texture)
    # Set the load action of the depth attachment descriptor to clear
    depthAttachmentDescriptor.setLoadAction_(Metal.MTLLoadActionClear)
    # Set the store action of the depth attachment descriptor to store
    depthAttachmentDescriptor.setStoreAction_(Metal.MTLStoreActionStore)
    # Set the clear depth of the depth attachment descriptor to 1.0
    depthAttachmentDescriptor.setClearDepth_(1.0)
    # Set the depth attachment of the render pass descriptor to the depth attachment descriptor
    renderPassDescriptor.setDepthAttachment_(depthAttachmentDescriptor)


    # Create a Metal vertex buffer (vertices and colors)
    vertices = np.array([
         [0.0, 0.5, 0.0, 1.0],
         [-0.5, -0.5, 0.0, 1.0],
         [0.5, -0.5, 0.0, 1.0]
    ], dtype = np.float32)
    colors = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
    ], dtype = np.float32)
    vertexData = np.concatenate((vertices, colors), axis = 1)
    vertexData = vertexData.flatten()


    vertexBuffer = device.newBufferWithBytes_length_options_(vertexData, vertexData.nbytes, Metal.MTLResourceStorageModeShared)


    # Preallocate the buffers that will be used to copy the texture data
    # We will copy the contents of the Metal texture (color) to an OpenCV image (numpy array)
    region = Metal.MTLRegionMake2D(0, 0, WIDTH, HEIGHT )
    bytesPerRow = 4 * WIDTH
    bytesPerImage = 4 * WIDTH * HEIGHT 
    # Calculate the size of the buffer in bytes
    buffer_size = bytesPerRow * region.size.height
    # Create the buffer
    buffer = ctypes.create_string_buffer(buffer_size)
    # make a numpy view of the buffer (without copying the data)
    image_view = np.ndarray(buffer=buffer, dtype=np.uint8, shape=(HEIGHT , WIDTH, 4))

    count = 1
    start_time = time.time()
    rotation = 0.0

    while(True):
        count += 1
        # Calculate FPS
        rotation += 0.01
        if rotation > 2 * np.pi:
            rotation = 0.0

        if count % 100 == 0:
            print("rendering: FPS: %f" % (count / (time.time() - start_time)))
            start_time = time.time()
            count = 1

        #To copy pixel data from system memory into the texture, call replace(region:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:) or replace(region:mipmapLevel:withBytes:bytesPerRow:).
        #color_texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(region, 0, 0, frame, bytesPerRow, bytesPerImage)



        # Create a perspective projection matrix
        projectionMatrix = np.array([
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]
        ], dtype = np.float32)
        

        # Create a random view matrix every frame.
        viewMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype = np.float32)


        # rotate around the y axis
        # viewMatrix[0, 0] = math.cos(rotation)
        # viewMatrix[0, 2] = math.sin(rotation)
        # viewMatrix[2, 0] = -math.sin(rotation)
        # viewMatrix[2, 2] = math.cos(rotation)

        # rotate around the x axis
        # viewMatrix[1, 1] = math.cos(rotation)
        # viewMatrix[1, 2] = -math.sin(rotation)
        # viewMatrix[2, 1] = math.sin(rotation)
        # viewMatrix[2, 2] = math.cos(rotation)

        # rotate around the z axis
        viewMatrix[0, 0] = math.cos(rotation)
        viewMatrix[0, 1] = -math.sin(rotation)
        viewMatrix[1, 0] = math.sin(rotation)
        viewMatrix[1, 1] = math.cos(rotation)




        # Create a Metal command buffer
        commandBuffer = commandQueue.commandBuffer()

        # Create a Metal render command encoder
        commandEncoder = commandBuffer.renderCommandEncoderWithDescriptor_(renderPassDescriptor)

        #print("renderCommandEncoder: ", renderCommandEncoder)
        commandEncoder.setRenderPipelineState_(renderPipeline)
        commandEncoder.setVertexBuffer_offset_atIndex_(vertexBuffer, 0, 0)
        commandEncoder.setVertexBytes_length_atIndex_(projectionMatrix.view(np.uint8), projectionMatrix.nbytes, 1)
        commandEncoder.setVertexBytes_length_atIndex_(viewMatrix.view(np.uint8), viewMatrix.nbytes, 2)

        commandEncoder.drawPrimitives_vertexStart_vertexCount_(Metal.MTLPrimitiveTypeTriangle, 0, 3)
        commandEncoder.endEncoding()

        # Commit the Metal command buffer
        commandBuffer.commit()
        # Wait for the command buffer to finish executing
        commandBuffer.waitUntilCompleted()

        # Copy the contents of the Metal texture to the buffer
        color_texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(buffer, bytesPerRow, region, 0)

        # the alpha channel is the depth channel, extract it
        depth = image_view[:, :, 3]
        # colorize the depth channel
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        # color channels are BGR, convert to RGB
        image = image_view[:, :, ::-1]

        # Display the OpenCV images
        cv2.imshow('image', image)
        cv2.imshow('depth', depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


