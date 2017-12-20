//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#include "imageutil.h"
#include "gpu_convert.h"
#include <iostream>
using namespace cv;
using namespace std;

cpu_image<uchar4> cpu_image_from_qimage(const QImage& image) {
    if (!image.isNull()) {
        QImage qi = image;
        if (qi.format() != QImage::Format_RGB32) {
            qi = qi.convertToFormat(QImage::Format_RGB32);
        }
        return cpu_image<uchar4>((const uchar4*)qi.bits(), qi.bytesPerLine(), qi.width(), qi.height());
    }
    return cpu_image<uchar4>();
}


QImage cpu_data_to_qimage(const cpu_image_data *data) {
    if (!data) return QImage();

    if (data->type_id() == pixel_type_id<float>()) {
        QImage img(data->w(), data->h(), QImage::Format_Indexed8);
        for (unsigned i = 0; i < 256; ++i) img.setColor(i, 0xff000000 | (i << 16) | (i << 8) | i);
        const float *p = (const float*)data->ptr(); 
        for (unsigned j = 0; j < data->h(); ++j) {
            unsigned char* q = img.scanLine(j);
            for (unsigned i = 0; i < data->w(); ++i) {
                *q++ = (unsigned char)(255.0f * clamp(*p++, 0.0f, 1.0f));
            }
        }
        return img;
    } 

    if (data->type_id() == pixel_type_id<unsigned char>()) {
        QImage img(data->w(), data->h(), QImage::Format_Indexed8);
        for (unsigned i = 0; i < 256; ++i) img.setColor(i, 0xff000000 | (i << 16) | (i << 8) | i);
        const unsigned char *p = (const unsigned char*)data->ptr(); 
        for (unsigned j = 0; j < data->h(); ++j) {
            unsigned char* q = img.scanLine(j);
            for (unsigned i = 0; i < data->w(); ++i) {
                *q++ = *p++;
            }
        }
        return img;
    } 
    
    if (data->type_id() == pixel_type_id<float4>()) {
        QImage img(data->w(), data->h(), QImage::Format_RGB32);
        const unsigned N = data->w() * data->h();
        const float4 *p = (const float4*)data->ptr(); 
        uchar4* q = (uchar4*)img.bits();
        for (unsigned i = 0; i < N; ++i) {
            q->x = (unsigned char)(255.0f * clamp(p->x, 0.0f, 1.0f));
            q->y = (unsigned char)(255.0f * clamp(p->y, 0.0f, 1.0f));
            q->z = (unsigned char)(255.0f * clamp(p->z, 0.0f, 1.0f));
            q->w = 255;
            p++;
            q++;
        }
        return img;
    }
    
    if (data->type_id() == pixel_type_id<uchar4>()) {
        QImage img(data->w(), data->h(), QImage::Format_RGB32);
        const unsigned N = data->w() * data->h();
        const uchar4 *p = (const uchar4*)data->ptr(); 
        uchar4* q = (uchar4*)img.bits();
        for (unsigned i = 0; i < N; ++i) {
            q->x = p->x;
            q->y = p->y;
            q->z = p->z;
            q->w = 255;
            p++;
            q++;
        }
        return img;
    }

    return QImage();
}


template <> gpu_image<uchar4> gpu_image_from_qimage(const QImage& image) {

    if (!image.isNull()) {
        QImage image4 = image.convertToFormat(QImage::Format_RGB32);
        return gpu_image<uchar4>((uchar4*)image4.bits(), image4.bytesPerLine(), image4.width(), image4.height());
    }
    return gpu_image<uchar4>();
}

template <> gpu_image<float4> gpu_image_from_qimage(const QImage& image) {
    if (!image.isNull()) {
        return gpu_8u_to_32f(gpu_image_from_qimage<uchar4>(image));
    }
    return gpu_image<float4>();
}


QImage gpu_image_to_qimage(const gpu_image<uchar>& image) {
    QImage dst(image.w(), image.h(), QImage::Format_Indexed8);
    dst.setColorCount(256);
    for (int i = 0; i < 256; ++i) dst.setColor(i, qRgb(i,i,i));
    copy(dst.bits(), dst.bytesPerLine(), &image);
    return dst;
}


QImage gpu_image_to_qimage(const gpu_image<uchar4>& image) {
    QImage dst(image.w(), image.h(), QImage::Format_RGB32);
    copy(dst.bits(), dst.bytesPerLine(), &image);
    return dst;
}


QImage gpu_image_to_qimage(const gpu_image<float>& image) {
    return gpu_image_to_qimage(gpu_32f_to_8u(image));
}

QImage gpu_image_to_qimage(const gpu_image<float4>& image) {
    return gpu_image_to_qimage(gpu_32f_to_8u(image));
}

void qimage_to_mat(const QImage& image, cv::OutputArray out) {

    switch(image.format()) {
        case QImage::Format_Invalid:
        {
            Mat empty;
            empty.copyTo(out);
            break;
        }
        case QImage::Format_RGB32:
        {
            Mat view(image.height(),image.width(),CV_8UC4,(void *)image.constBits(),image.bytesPerLine());
            view.copyTo(out);
            break;
        }
        case QImage::Format_RGB888:
        {
            Mat view(image.height(),image.width(),CV_8UC3,(void *)image.constBits(),image.bytesPerLine());
            cvtColor(view, out, COLOR_RGB2BGR);
            break;
        }
        default:
        {
            QImage conv = image.convertToFormat(QImage::Format_ARGB32);
            Mat view(conv.height(),conv.width(),CV_8UC4,(void *)conv.constBits(),conv.bytesPerLine());
            view.copyTo(out);
            break;
        }
    }
}

void mat_to_qimage(cv::InputArray image, QImage& out)
{
    switch(image.type())
    {
        case CV_8UC4:
        {
            Mat view(image.getMat());
            QImage view2(view.data, view.cols, view.rows, view.step[0], QImage::Format_ARGB32);
            out = view2.copy();
            break;
        }
        case CV_8UC3:
        {
            Mat mat;
            cvtColor(image, mat, COLOR_BGR2BGRA); //COLOR_BGR2RGB doesn't behave so use RGBA
            QImage view(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_ARGB32);
            out = view.copy();
            break;
        }
        case CV_8UC1:
        {
            Mat mat;
            cvtColor(image, mat, COLOR_GRAY2BGRA);
            QImage view(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_ARGB32);
            out = view.copy();
            break;
        }
        default:
        {
            throw invalid_argument("Image format not supported");
            break;
        }
    }
}

template <> gpu_image<uchar4> gpu_image_from_mat(cv::InputArray image) {

    if (image.empty()) 
        gpu_image<uchar4>();

    switch(image.type())
    {
        case CV_8UC4:
        {
            Mat view(image.getMat());
            return gpu_image<uchar4>((uchar4*)view.data, view.step[0], view.cols, view.rows);
        }
        case CV_8UC3:
        {
            Mat mat;
            cvtColor(image, mat, COLOR_BGR2BGRA); //COLOR_BGR2RGB doesn't behave so use RGBA
            return gpu_image<uchar4>((uchar4*)mat.data, mat.step[0], mat.cols, mat.rows);
        }
        case CV_8UC1:
        {
            Mat mat;
            cvtColor(image, mat, COLOR_GRAY2BGRA);
            return gpu_image<uchar4>((uchar4*)mat.data, mat.step[0], mat.cols, mat.rows);
        }
        default:
        {
            throw invalid_argument("Image format not supported");
            break;
        }
    }

    return gpu_image<uchar4>();
}

template <> gpu_image<float4> gpu_image_from_mat(cv::InputArray image) {
    if (image.empty())
        return gpu_image<float4>();
    if( image.type() == CV_32FC4)
    {
        Mat data(image.getMat());
        Mat scaled;
        data.convertTo(scaled, CV_32FC4, 1.0/255.0);
        return gpu_image<float4>((float4*)scaled.data, scaled.step[0], scaled.cols, scaled.rows);
    }
    return gpu_8u_to_32f(gpu_image_from_mat<uchar4>(image));
}

void gpu_image_to_mat(const gpu_image<uchar>& image, cv::OutputArray out) {

    Mat data(image.h(), image.w(), CV_8UC1);
    //for (int i = 0; i < 256; ++i) dst.setColor(i, qRgb(i,i,i));
    copy(data.data, data.step[0], &image);
    data.copyTo(out);
}

void gpu_image_to_mat(const gpu_image<uchar4>& image, cv::OutputArray out) {

    Mat data(image.h(), image.w(), CV_8UC4);
    copy(data.data, data.step[0], &image);
    data.copyTo(out);
}

void gpu_image_to_mat(const gpu_image<float>& image, cv::OutputArray out) {
    gpu_image_to_mat(gpu_32f_to_8u(image), out);
}

void gpu_image_to_mat(const gpu_image<float4>& image, cv::OutputArray out) {
    gpu_image_to_mat(gpu_32f_to_8u(image), out);
}

