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

#define PLAGIN_NAME "cudacurves"
#define PLAGIN_SHORT_DESCRIPTION "CUDA curves Filter"

GST_DEBUG_CATEGORY_STATIC (gst_plugin_template_debug);
#define GST_CAT_DEFAULT gst_plugin_template_debug

typedef struct _GstPlugincudacurves GstPlugincudacurves;
typedef struct _GstPlugincudacurvesClass GstPlugincudacurvesClass;

#define GST_TYPE_PLUGIN_TEMPLATE \
  (gst_plugin_template_get_type())
#define GST_PLUGIN_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_PLUGIN_TEMPLATE,GstPlugincudacurves))
#define GST_PLUGIN_TEMPLATE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_PLUGIN_TEMPLATE,GstPlugincudacurvesClass))
#define GST_IS_PLUGIN_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_PLUGIN_TEMPLATE))
#define GST_IS_PLUGIN_TEMPLATE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_PLUGIN_TEMPLATE))

struct _GstPlugincudacurves
{
  GstVideoFilter videofilter;

  gint width;
  gint height;

  // curve file names
  gchar* red;
  gchar* green;
  gchar* blue;

  // curves
  unsigned char curve[256][4];
  cudaArray* curves;
  cudaTextureObject_t curves_tex;
};

struct _GstPlugincudacurvesClass
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
  PROP_RED,
  PROP_GREEN,
  PROP_BLUE
};

/* debug category for fltering log messages
 */
#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_plugin_template_debug, PLAGIN_NAME, 0, PLAGIN_SHORT_DESCRIPTION);

GST_BOILERPLATE_FULL (GstPlugincudacurves, gst_plugin_template,
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
    "Curves",
    "Mikhail Durnev <mikhail_durnev@mentor.com>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_video_filter_sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_video_filter_src_template));
}

static void
gst_plugin_template_class_init (GstPlugincudacurvesClass * klass)
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

  g_object_class_install_property (gobject_class, PROP_RED,
      g_param_spec_string ("red", "Red", "Red = ",
          "", (GParamFlags)G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_GREEN,
      g_param_spec_string ("green", "Green", "Green = ",
          "", (GParamFlags)G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_BLUE,
      g_param_spec_string ("blue", "Blue", "Blue = ",
          "", (GParamFlags)G_PARAM_READWRITE));

  btrans_class->set_caps = gst_plugin_template_set_caps;
  btrans_class->transform = NULL;
  btrans_class->transform_ip = gst_plugin_template_filter_inplace;
}

static void
gst_plugin_template_init (GstPlugincudacurves * plugin_template,
    GstPlugincudacurvesClass * g_class)
{
  GST_DEBUG ("init");

  plugin_template->red   = NULL;
  plugin_template->green = NULL;
  plugin_template->blue  = NULL;

  plugin_template->curves = NULL;
  plugin_template->curves_tex = 0;
}

static void
gst_plugin_template_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstPlugincudacurves *filter = GST_PLUGIN_TEMPLATE (object);

  GST_OBJECT_LOCK (filter);
  switch (prop_id) {
    case PROP_RED:
        filter->red = (gchar*)malloc(strlen(g_value_get_string (value)) + 1);
        strcpy(filter->red, g_value_get_string (value));
        GST_DEBUG("red = %s\n", filter->red);
        break;
    case PROP_GREEN:
        filter->green = (gchar*)malloc(strlen(g_value_get_string (value)) + 1);
        strcpy(filter->green, g_value_get_string (value));
        GST_DEBUG("green = %s\n", filter->green);
        break;
    case PROP_BLUE:
        filter->blue = (gchar*)malloc(strlen(g_value_get_string (value)) + 1);
        strcpy(filter->blue, g_value_get_string (value));
        GST_DEBUG("red = %s\n", filter->blue);
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
  GstPlugincudacurves *filter = GST_PLUGIN_TEMPLATE (object);

  GST_OBJECT_LOCK (filter);
  switch (prop_id) {
    case PROP_RED:
        g_value_set_string (value, filter->red);
        break;
    case PROP_GREEN:
        g_value_set_string (value, filter->green);
        break;
    case PROP_BLUE:
        g_value_set_string (value, filter->blue);
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
  GstPlugincudacurves *filter = GST_PLUGIN_TEMPLATE (object);

  if (filter->red)
  {
      free(filter->red);
      filter->red = NULL;
  }

  if (filter->green)
  {
      free(filter->green);
      filter->green = NULL;
  }

  if (filter->blue)
  {
      free(filter->blue);
      filter->blue = NULL;
  }

  if (filter->curves != NULL)
  {
      CUDA_CHECK_RETURN(cudaFreeArray(filter->curves));
      CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->curves_tex));
  }

  //G_OBJECT_CLASS (object)->finalize (object);

  GST_DEBUG("finalize");
}

//=============================================================================================================================
// Curve calculation
//=============================================================================================================================
typedef struct matrices {
   float* x;
   float* y;
   float* h;
   float* l;
   float* delta;
   float* lambda;
   float* c;
   float* d;
   float* b;
} Matrices;

static void prepmatrix(int* curve, int N, Matrices* m)
{
   int i = 0, k = 0;
   for (i = 0; i < 256; i++)
   {
       if (curve[i] >= 0)
       {
           m->x[k] = i;
           m->y[k++] = curve[i];
       }
   }
}

static void allocmatrix(int N, Matrices* m)
{
   //allocate memory for matrices
   m->x = (float*)malloc(N * sizeof(float));
   m->y = (float*)malloc(N * sizeof(float));
   m->h = (float*)malloc(N * sizeof(float));
   m->l = (float*)malloc(N * sizeof(float));
   m->delta = (float*)malloc(N * sizeof(float));
   m->lambda = (float*)malloc(N * sizeof(float));
   m->c = (float*)malloc(N * sizeof(float));
   m->d = (float*)malloc(N * sizeof(float));
   m->b = (float*)malloc(N * sizeof(float));
}

static void freematrix(Matrices* m)
{
   free(m->x);
   free(m->y);
   free(m->h);
   free(m->l);
   free(m->delta);
   free(m->lambda);
   free(m->c);
   free(m->d);
   free(m->b);
}

static int calc_curve(gchar* filename, unsigned char res[256][4], int idx)
{
    int k = 0, i = 0;

    FILE* file=NULL;
    long  filesize = 0;
    char* filedata = NULL;

    int N = 0;     //N - number of data points

    int curve[256];

    Matrices m;

    if (filename == NULL)
    {
        // default curve
        GST_DEBUG("Default curve");

        for (i = 0; i < 256; i++)
        {
            res[i][idx] = (unsigned char)i;
        }

        return TRUE;
    }

    // Load curve from a file
    file = fopen(filename, "rt");
    if (!file)
    {
        GST_DEBUG("File not found: %s", filename);
        return FALSE;
    }

    fseek(file, 0, SEEK_END);
    filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (filesize <= 0 ||
        (filedata = (char*)malloc(filesize)) == NULL ||
        fread(filedata, 1, filesize, file) != filesize)
    {
        GST_DEBUG("File read error");
        return FALSE;
    }
    fclose(file);

    for (i = 0; i < 256; i++)
    {
        curve[i] = -1;
    }

    char* line = filedata;
    for (i = 0; i < filesize; i++)
    {
        if (filedata[i] == '\n' || i >= filesize)
        {
            int x, y;
            sscanf(line, "%i %i", &x, &y);
            line = &filedata[i + 1];

            if (x >= 0 && x < 256 && y >= 0 && y < 256 && curve[x] == -1)
            {
                curve[x] = y;
                ++N;
            }
            else
            {
                GST_DEBUG("Error in input data");
                return FALSE;
            }
        }
    }

    allocmatrix(N, &m);
    prepmatrix(curve, N, &m);

    --N;

    for (k = 1; k <= N; k++)
    {
        m.h[k] = m.x[k] - m.x[k-1];
        if (m.h[k] == 0)
        {
            GST_DEBUG("Error, x[%d]=x[%d]", k, k-1);
            return FALSE;
        }
        m.l[k] = (m.y[k] - m.y[k-1])/m.h[k];
    }
    m.delta[1] = - m.h[2]/(2*(m.h[1]+m.h[2]));
    m.lambda[1] = 1.5*(m.l[2] - m.l[1])/(m.h[1]+m.h[2]);
    for (k = 3; k <= N; k++)
    {
       m.delta[k-1] = - m.h[k]/(2*m.h[k-1] + 2*m.h[k] + m.h[k-1]*m.delta[k-2]);
       m.lambda[k-1] = (3*m.l[k] - 3*m.l[k-1] - m.h[k-1]*m.lambda[k-2]) /
                     (2*m.h[k-1] + 2*m.h[k] + m.h[k-1]*m.delta[k-2]);
    }
    m.c[0] = 0;
    m.c[N] = 0;
    for (k = N; k >= 2; k--)
    {
       m.c[k-1] = m.delta[k-1]*m.c[k] + m.lambda[k-1];
    }
    for (k = 1; k <= N; k++)
    {
       m.d[k] = (m.c[k] - m.c[k-1])/(3*m.h[k]);
       m.b[k] = m.l[k] + (2*m.c[k]*m.h[k] + m.h[k]*m.c[k-1])/3;
    }

    int s;
    for (s = 0; s <= 255; s++)
    {
        //find k, where s in [x_k-1; x_k]
        int k;
        for (k = 1; k <= N; k++)
        {
            if (s >= m.x[k-1] && s <= m.x[k])
            {
                break;
            }
        }
        float F = m.y[k] + m.b[k]*(s-m.x[k]) + m.c[k]*pow(s-m.x[k], 2) + m.d[k]*pow(s-m.x[k], 3);
        res[s][idx] = (unsigned char)F;
    }

    freematrix(&m);

    return TRUE;
}

static int calc_curves(GstPlugincudacurves* filter)
{
    int res = calc_curve(filter->red, filter->curve, 2);
    res &= calc_curve(filter->green, filter->curve, 1);
    res &= calc_curve(filter->blue, filter->curve, 0);

    if (res)
    {
        if (filter->curves != NULL)
        {
            CUDA_CHECK_RETURN(cudaFreeArray(filter->curves));
            CUDA_CHECK_RETURN(cudaDestroyTextureObject(filter->curves_tex));
        }

        cudaChannelFormatDesc desci = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);

        CUDA_CHECK_RETURN(cudaMallocArray(&filter->curves, &desci, 256, 1));
        CUDA_CHECK_RETURN(cudaMemcpyToArray(filter->curves, 0, 0, filter->curve, 256*4, cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = filter->curves;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;

        CUDA_CHECK_RETURN(cudaCreateTextureObject(&filter->curves_tex, &resDesc, &texDesc, NULL));
    }

    return res;
}
//=============================================================================================================================


static gboolean
gst_plugin_template_set_caps (GstBaseTransform * bt,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstPlugincudacurves *plugin_template;
  GstStructure *structure = NULL;
  gboolean ret = FALSE;

  plugin_template = GST_PLUGIN_TEMPLATE (bt);

  structure = gst_caps_get_structure (incaps, 0);

  GST_OBJECT_LOCK (plugin_template);
  if (gst_structure_get_int (structure, "width", &plugin_template->width) &&
      gst_structure_get_int (structure, "height", &plugin_template->height))
  {
    /* Check width and height and modify other plugin_template members accordingly */

    ret = calc_curves(plugin_template);
  }
  GST_OBJECT_UNLOCK (plugin_template);

  return ret;
}

__global__ void video_filter(cudaTextureObject_t in, uchar4* out, size_t pitch, int width, int height,
                                cudaTextureObject_t curves_tex)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ uchar4 curve[256];

    curve[tid] = tex2D<uchar4>(curves_tex, tid, 0);
    __syncthreads();

    uchar4 v = tex2D<uchar4>(in, x, y);

    v.x = curve[v.x].x;
    v.y = curve[v.y].y;
    v.z = curve[v.z].z;

    *(uchar4*)((char*)out + (pitch * y + x * sizeof(uchar4))) = v;
}

static GstFlowReturn
gst_plugin_template_filter_inplace (GstBaseTransform * base_transform,
    GstBuffer * buf)
{
  GstPlugincudacurves *plugin_template = GST_PLUGIN_TEMPLATE (base_transform);
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
                                      plugin_template->curves_tex);

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
