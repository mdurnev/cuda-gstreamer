/*
 * GStreamer
 * Copyright (C) <1999> Erik Walthinsen <omega@cse.ogi.edu>
 * Copyright (C) <2003> David Schleef <ds@schleef.org>
 * Copyright (C) <2012> Mikhail Durnev <mdurnev@rhonda.ru>
 * Copyright (C) <2014> Mikhail Durnev <mikhail_durnev@mentor.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-plugin
 *
 * FIXME:Describe plugin here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m videotestsrc ! plugin ! autovideosink
 * ]|
 * </refsect2>
 */
 
#ifdef HAVE_CONFIG_H
#include "../../common/config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <string.h>

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t stat = value;                                        \
    if (stat != cudaSuccess) {                                       \
        GST_DEBUG("Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(stat), __LINE__, __FILE__);       \
    } }

typedef unsigned int uint32_t;

#define PLAGIN_NAME "cudalens"
#define PLAGIN_SHORT_DESCRIPTION "CUDA lens Filter"

GST_DEBUG_CATEGORY_STATIC (gst_plugin_template_debug);
#define GST_CAT_DEFAULT gst_plugin_template_debug

typedef struct _GstPlugincudalens GstPlugincudalens;
typedef struct _GstPlugincudalensClass GstPlugincudalensClass;

#define GST_TYPE_PLUGIN_TEMPLATE \
  (gst_plugin_template_get_type())
#define GST_PLUGIN_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_PLUGIN_TEMPLATE,GstPlugincudalens))
#define GST_PLUGIN_TEMPLATE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_PLUGIN_TEMPLATE,GstPlugincudalensClass))
#define GST_IS_PLUGIN_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_PLUGIN_TEMPLATE))
#define GST_IS_PLUGIN_TEMPLATE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_PLUGIN_TEMPLATE))

struct _GstPlugincudalens
{
  GstVideoFilter videofilter;

  gint width;
  gint height;

  gfloat factor;

  // Barrel distortion compensation matrix
  cudaArray* barrel_idx;
  cudaTextureObject_t barrel_idx_tex;
  cudaArray* afterpoint;
  cudaTextureObject_t afterpoint_tex;
};

struct _GstPlugincudalensClass
{
  GstVideoFilterClass parent_class;
};


enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_FACTOR
};

/* debug category for fltering log messages
 */
#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_plugin_template_debug, PLAGIN_NAME, 0, PLAGIN_SHORT_DESCRIPTION);

GST_BOILERPLATE_FULL (GstPlugincudalens, gst_plugin_template,
    GstVideoFilter, GST_TYPE_VIDEO_FILTER, DEBUG_INIT);

static void gst_plugin_template_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_plugin_template_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_plugin_template_finalize (GObject * object);

static gboolean gst_plugin_template_set_caps (GstBaseTransform * bt,
    GstCaps * incaps, GstCaps * outcaps);
//static GstFlowReturn gst_plugin_template_filter (GstBaseTransform * bt,
//    GstBuffer * outbuf, GstBuffer * inbuf);
static GstFlowReturn
gst_plugin_template_filter_inplace (GstBaseTransform * base_transform,
    GstBuffer * buf);

#define ALLOWED_CAPS_STRING \
    GST_VIDEO_CAPS_BGRx

static GstStaticPadTemplate gst_video_filter_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (ALLOWED_CAPS_STRING)
    );

static GstStaticPadTemplate gst_video_filter_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (ALLOWED_CAPS_STRING)
    );

/* GObject method implementations */

static void
gst_plugin_template_base_init (gpointer klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstVideoFilterClass *videofilter_class = GST_VIDEO_FILTER_CLASS (klass);
  GstCaps *caps;

  gst_element_class_set_details_simple (element_class,
    PLAGIN_NAME,
    "Filter/Effect/Video",
    "Removes fisheye",
    "Mikhail Durnev <mikhail_durnev@mentor.com>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_video_filter_sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_video_filter_src_template));
}

static void
gst_plugin_template_class_init (GstPlugincudalensClass * klass)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *btrans_class;
  GstVideoFilterClass *video_filter_class;

  gobject_class = (GObjectClass *) klass;
  btrans_class = (GstBaseTransformClass *) klass;
  video_filter_class = (GstVideoFilterClass *) klass;

  gobject_class->set_property = gst_plugin_template_set_property;
  gobject_class->get_property = gst_plugin_template_get_property;
  gobject_class->finalize = gst_plugin_template_finalize;

  g_object_class_install_property (gobject_class, PROP_FACTOR,
      g_param_spec_float ("factor", "Factor", "Factor = ",
          0.0, 0.00001, 0.0000008, (GParamFlags)G_PARAM_READWRITE));

  btrans_class->set_caps = gst_plugin_template_set_caps;
  btrans_class->transform = NULL;
  btrans_class->transform_ip = gst_plugin_template_filter_inplace;
}

static void
gst_plugin_template_init (GstPlugincudalens * plugin_template,
    GstPlugincudalensClass * g_class)
{
  GST_DEBUG ("init");

  plugin_template->factor = 0.0000008;
  plugin_template->barrel_idx = NULL;
  plugin_template->afterpoint = NULL;
  plugin_template->barrel_idx_tex = 0;
  plugin_template->afterpoint_tex = 0;
}

static void
gst_plugin_template_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstPlugincudalens *filter = GST_PLUGIN_TEMPLATE (object);

  GST_OBJECT_LOCK (filter);
  switch (prop_id) {
    case PROP_FACTOR:
        filter->factor = g_value_get_float (value);
        GST_DEBUG("factor = %.8f\n", (double)filter->factor);
        break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (filter);
}

static void
gst_plugin_template_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstPlugincudalens *filter = GST_PLUGIN_TEMPLATE (object);

  GST_OBJECT_LOCK (filter);
  switch (prop_id) {
    case PROP_FACTOR:
        g_value_set_float (value, filter->factor);
        break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (filter);
}

static void
gst_plugin_template_finalize (GObject * object)
{
  GstPlugincudalens *filter = GST_PLUGIN_TEMPLATE (object);

  if (filter->barrel_idx != NULL)
  {
      CUDA_CHECK_RETURN(cudaFreeArray(filter->barrel_idx));
      CUDA_CHECK_RETURN(cudaFreeArray(filter->afterpoint));
      CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->barrel_idx_tex));
      CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->afterpoint_tex));
  }

  //G_OBJECT_CLASS (object)->finalize (object);

  GST_DEBUG("finalize");
}

__global__ void fill_matrix(uint2* barrel_idx, size_t pitch1, float2* afterpoint, size_t pitch2, float factor, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    double k = (double)height / width / 3 * 4;
    k *= k;

    // move coordinate center to the center of the frame
    int x1 = x - (width >> 1);
    int y1 = y - (height >> 1);

    // compensate barrel distortion of the lens
    double d = 1 - ((x1 * x1) * k + y1 * y1) * factor;

    // calculate new coordinates
    double fx1 = x1 * d + (width >> 1);
    double fy1 = y1 * d + (height >> 1);

    // bilinear interpolation
    x1 = (int)fx1;
    y1 = (int)fy1;

    (*(float2*)((char*)afterpoint + (pitch2 * y + x * sizeof(float2)))).x = (float)(fx1 - x1);
    (*(float2*)((char*)afterpoint + (pitch2 * y + x * sizeof(float2)))).y = (float)(fy1 - y1);
    (*(uint2*)((char*)barrel_idx + (pitch1 * y + x * sizeof(uint2)))).x = x1;
    (*(uint2*)((char*)barrel_idx + (pitch1 * y + x * sizeof(uint2)))).y = y1;
}

static void calc_matrix(GstPlugincudalens* filter, int stride)
{
    int width = filter->width;
    int height = filter->height;

    GST_DEBUG("width=%d, height=%d\n", width, height);

    if (filter->barrel_idx != NULL)
    {
        CUDA_CHECK_RETURN(cudaFreeArray(filter->barrel_idx));
        CUDA_CHECK_RETURN(cudaFreeArray(filter->afterpoint));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->barrel_idx_tex));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->afterpoint_tex));
    }

    cudaChannelFormatDesc desci = cudaCreateChannelDesc<uint2>();
    cudaChannelFormatDesc descf = cudaCreateChannelDesc<float2>();

    CUDA_CHECK_RETURN(cudaMallocArray(&filter->barrel_idx, &desci, width, height));
    CUDA_CHECK_RETURN(cudaMallocArray(&filter->afterpoint, &descf, width, height));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = filter->barrel_idx;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    CUDA_CHECK_RETURN(cudaCreateTextureObject(&filter->barrel_idx_tex, &resDesc, &texDesc, NULL));

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.res.array.array = filter->afterpoint;

    CUDA_CHECK_RETURN(cudaCreateTextureObject(&filter->afterpoint_tex, &resDesc, &texDesc, NULL));

    size_t pitch1, pitch2;
    void* barrel_idx;
    void* afterpoint;

    CUDA_CHECK_RETURN(cudaMallocPitch(&barrel_idx, &pitch1, width * sizeof(uint2), height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&afterpoint, &pitch2, width * sizeof(float2), height));

    // fill barrel distortion matrix
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    fill_matrix<<<dimGrid, dimBlock>>>((uint2*)barrel_idx, pitch1, (float2*)afterpoint, pitch2, filter->factor, width, height);

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaMemcpy2DToArray(filter->barrel_idx, 0, 0, barrel_idx, pitch1, width * sizeof(uint2), height, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2DToArray(filter->afterpoint, 0, 0, afterpoint, pitch2, width * sizeof(float2), height, cudaMemcpyDeviceToDevice));
}

static gboolean
gst_plugin_template_set_caps (GstBaseTransform * bt,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstPlugincudalens *plugin_template;
  GstStructure *structure = NULL;
  gboolean ret = FALSE;

  plugin_template = GST_PLUGIN_TEMPLATE (bt);

  structure = gst_caps_get_structure (incaps, 0);

  GST_OBJECT_LOCK (plugin_template);
  if (gst_structure_get_int (structure, "width", &plugin_template->width) &&
      gst_structure_get_int (structure, "height", &plugin_template->height))
  {
    /* Check width and height and modify other plugin_template members accordingly */

    /* Calculate distortion compensation matrix */
    calc_matrix(plugin_template, plugin_template->width * 4);

    ret = TRUE;
  }
  GST_OBJECT_UNLOCK (plugin_template);

  return ret;
}

__global__ void video_filter(cudaTextureObject_t in, uchar4* out, size_t pitch, int width, int height,
                                cudaTextureObject_t barrel_idx_tex, cudaTextureObject_t afterpoint_tex)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    uint2 c = tex2D<uint2>(barrel_idx_tex, x, y);

    uchar4 v1 = tex2D<uchar4>(in, c.x, c.y);
    uchar4 v2 = tex2D<uchar4>(in, c.x, c.y + 1);
    uchar4 v3 = tex2D<uchar4>(in, c.x + 1, c.y);
    uchar4 v4 = tex2D<uchar4>(in, c.x + 1, c.y + 1);

    float2 f = tex2D<float2>(afterpoint_tex, x, y);

    float4 a, b;
    a.x = v1.x + (v3.x - v1.x) * f.x;
    a.y = v1.y + (v3.y - v1.y) * f.x;
    a.z = v1.z + (v3.z - v1.z) * f.x;
    a.w = v1.w + (v3.w - v1.w) * f.x;

    b.x = v2.x + (v4.x - v2.x) * f.x;
    b.y = v2.y + (v4.y - v2.y) * f.x;
    b.z = v2.z + (v4.z - v2.z) * f.x;
    b.w = v2.w + (v4.w - v2.w) * f.x;

    uchar4 v;
    v.x = (unsigned char)(a.x + (b.x - a.x) * f.y);
    v.y = (unsigned char)(a.y + (b.y - a.y) * f.y);
    v.z = (unsigned char)(a.z + (b.z - a.z) * f.y);
    v.w = (unsigned char)(a.w + (b.w - a.w) * f.y);

    *(uchar4*)((char*)out + (pitch * y + x * sizeof(uchar4))) = v;
}

static GstFlowReturn
gst_plugin_template_filter_inplace (GstBaseTransform * base_transform,
    GstBuffer * buf)
{
  GstPlugincudalens *plugin_template = GST_PLUGIN_TEMPLATE (base_transform);
  GstVideoFilter *videofilter = GST_VIDEO_FILTER (base_transform);

  gint width = plugin_template->width;
  gint height = plugin_template->height;
  gint stride = width * 4;

  unsigned long long *in = (unsigned long long *) GST_BUFFER_DATA (buf);
  /*
   * in[0] - device pointer to the allocated memory
   * in[1] - pitch in bytes
   * in[2] - texture object
   * in[3] - device memory allocated for image processing
   * in[4] - pitch in bytes
   * in[5] - texture object
   */

  dim3 dimBlock(16, 16);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  video_filter<<<dimGrid, dimBlock>>>((cudaTextureObject_t)in[2], (uchar4*)in[3], (size_t)in[4], width, height,
                                      plugin_template->barrel_idx_tex, plugin_template->afterpoint_tex);

  CUDA_CHECK_RETURN(cudaThreadSynchronize());
  CUDA_CHECK_RETURN(cudaGetLastError());

  // Swap buffers
  int i;
  for (i = 0; i < 3; i++)
  {
      unsigned long long x = in[i];
      in[i] = in[i + 3];
      in[i + 3] = x;
  }

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, PLAGIN_NAME, GST_RANK_NONE,
      GST_TYPE_PLUGIN_TEMPLATE);
}

/* gstreamer looks for this structure to register plugins
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    PLAGIN_NAME,
    PLAGIN_SHORT_DESCRIPTION,
    plugin_init,
    VERSION, "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
);

void test_plugin()
{
    GstPlugincudalens data;
    data.width = 640;
    data.height = 480;
    data.barrel_idx = NULL;
    data.afterpoint = NULL;
    data.barrel_idx_tex = 0;
    data.afterpoint_tex = 0;
    data.factor = 0.0000008;

    int stride = data.width * 4;

    calc_matrix(&data, stride);

    void *in = NULL;
    size_t in_pitch;
    CUDA_CHECK_RETURN(cudaMallocPitch(&in, &in_pitch, stride, data.height));

    void *out = NULL;
    size_t out_pitch;
    CUDA_CHECK_RETURN(cudaMallocPitch(&out, &out_pitch, stride, data.height));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = in;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.pitchInBytes = in_pitch;
    resDesc.res.pitch2D.width = stride;
    resDesc.res.pitch2D.height = data.height;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK_RETURN(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((data.width + dimBlock.x - 1) / dimBlock.x,
                 (data.height + dimBlock.y - 1) / dimBlock.y);

    video_filter<<<dimGrid, dimBlock>>>(tex, (uchar4*)out, out_pitch, data.width, data.height,
                                        data.barrel_idx_tex, data.afterpoint_tex);

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaDestroyTextureObject(tex));
    CUDA_CHECK_RETURN(cudaFree(in));
    CUDA_CHECK_RETURN(cudaFree(out));

    CUDA_CHECK_RETURN(cudaFreeArray(data.barrel_idx));
    CUDA_CHECK_RETURN(cudaFreeArray(data.afterpoint));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(data.barrel_idx_tex));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(data.afterpoint_tex));
}
