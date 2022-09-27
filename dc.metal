//
//  DomainColoring.metal
//
//  Created by asd on 01/04/2019.
//  Copyright © 2019 voicesync. All rights reserved.
// xcrun -sdk macosx metal -c dc.metal -o dc.air
// xcrun -sdk macosx metallib dc.air -o dc.metallib

#include <metal_stdlib>
using namespace metal;

constant const float E = 2.7182818284590452353602874713527;

typedef struct { uint x, y, width, height; } Geo;
typedef uint32_t color; // aa bb gg rr  32 bit color
typedef uint8_t byte;

template<typename T>
struct Complex
{
    T re=0, im=0;

    inline Complex() { re = im = 0; }
    inline Complex(T re) : re(re), im(0) { }
    inline Complex(T re, T im) : re(re), im(im) { }

    inline T arg() const { return ::atan2(im, re);  }
    inline T abs() const { return ::sqrt(sqmag());  }
    inline T sqmag() const {  return re*re + im*im;  }

    inline void operator=(const thread Complex &other) {
        re=other.re; im=other.im;
    }

    inline Complex  operator*(const thread Complex & other) const {
        return Complex(re*other.re - im*other.im,
                       re*other.im + im*other.re);
    }
    inline Complex  operator/(const thread Complex &other) const {
        T div=(other.re*other.re) + (other.im*other.im);
        Complex tmp;
        tmp.re=(re*other.re)+(im*other.im);
        tmp.re/=div;
        tmp.im=(im*other.re)-(re*other.im);
        tmp.im/=div;
        return tmp;
    }
    inline Complex  operator+(const thread Complex & other)  const {
        return Complex(re + other.re, im + other.im);
    }

    inline Complex  operator-(const thread  Complex & other)  const {
        return Complex(re - other.re, im - other.im);
    }

    inline thread Complex&  operator+=(const thread Complex &other) {
        re += other.re; im += other.im;
        return *this;
    }
    inline thread Complex&  operator-=(const thread Complex &other) {
        re -= other.re; im -= other.im;
        return *this;
    }
    inline thread Complex&  operator*=(const thread Complex &other) {
        auto _re=re*other.re - im*other.im;
        auto _im=re*other.im + im*other.re;
        re=_re; im=_im;
        return *this;
    }
    inline thread Complex&  operator/=(const thread Complex &other) {
        T div=(other.re*other.re) + (other.im*other.im);
        auto _re=(re*other.re)+(im*other.im);
        _re/=div;
        auto _im=(im*other.re)-(re*other.im);
        _im/=div;
        re=_re; im=_im;
        return *this;
    }

    inline Complex  operator*(const thread T& c) const {   return Complex(re * c, im * c); }
    inline Complex  operator+(const thread T& c) const {   return Complex(re + c, im);    }
    inline Complex  operator-(const thread T& c) const {   return Complex(re - c, im);    }
    inline Complex  operator/(const thread T& c) const {   return Complex(re / c, im / c); }
    inline Complex  operator-() const { return Complex(-re, -im); }

    inline Complex pow2() const { return *this * *this; }
    inline Complex pow3() const { return *this * *this * *this; }
    inline Complex pow4() const { return *this * *this * *this * *this; }

    Complex pow(unsigned n) const {
        switch(n) {
            case 0: return Complex(1,0);
            case 1: return *this;
            case 2: return pow2();
            case 3: return pow3();
            case 4: return pow4();
            default: //  > 4
                auto z=pow4();
                for (unsigned i=4; i<n; i++) z *= *this;
                return z;
        }
    }
    Complex pow(float n) const { // (𝑎+𝑖𝑏)𝑁=𝑟𝑁(cos(𝑁𝜃)+𝑖sin(𝑁𝜃))
        T rn=::pow(abs(), n), na=n*arg();
        return Complex(rn * ::cos(na), rn * ::sin(na));
    }
    Complex pow(Complex z) { // http://mathworld.wolfram.com/ComplexExponentiation.html
        // (a+bi)^(c+di)=(a^2+b^2)^(c/2)e^(-d * arg(a+ib)) × { cos[c arg(a+ib)+1/2dln(a^2+b^2)] + i sin[c arg(a+ib)+1/2 d ln(a^2+b^2)]}.

        T c=z.re, d=z.im;
        T m = ::pow(sqmag(), c/2) * ::exp(-d * arg());
        T _re = m * ::cos(c * arg() + 1/2 * d * ::log(sqmag()));
        T _im = m * ::sin(c * arg() + 1/2 * d * ::log(sqmag()));
        return Complex(_re, _im);
    }
    inline Complex sqrt() const {
        T a=abs();
        return Complex(::sqrt((a+re)/2), ::sign(im) * ::sqrt((a-re)/2) );
    }
    inline Complex log() const {
        return Complex(::log(abs()), arg());
    }
    inline Complex cosh() const {
        const T x = this->re, y = this->im;
        return Complex(::cosh(x) * ::cos(y), ::sinh(x) * ::sin(y));
    }
    inline Complex sinh() const {
        const T x = this->re, y = this->im;
        return Complex(::sinh(x) * ::cos(y), ::cosh(x) * ::sin(y));
    }
    inline Complex sin() const {
        const T x = this->re, y = this->im;
        return Complex(::sin(x) * ::cosh(y),  ::cos(x) * ::sinh(y));
    }
    inline Complex cos() const {
        const T x = this->re, y = this->im;
        return Complex(::cos(x) * ::cosh(y), -::sin(x) * ::sinh(y));
    }
    inline Complex tan() const {
        return sin()/cos();
    }
    inline Complex acos() const {
        const Complex t = asin();
        const T __pi_2 = 1.7514;
        return Complex(__pi_2 - t.re, -t.im);
    }
    inline Complex asin() const {
        Complex t(-im, re);
        t = t.asinh();
        return Complex(t.im, -t.re);
    }
    inline Complex atan() const { // atan(Z) = 0.5 atan(2x, 1 - x2 - y2) + 0.25 i alog((x2 + (y+1)2)/(x2 + (y-1)2))
        return Complex(
            0.50 * ::atan2(2*re, 1 - re*re - im*im) ,
            0.25 * ::log((re*re + (im+1)*(im+1))/(re*re + (im-1)*(im-1))));
    }
    inline Complex asinh() const {
        Complex t( (re-im) * (re+im)+1, 2*re*im);
        t = t.sqrt();
        return (t + *this).log();
    }
};

typedef Complex<float> ComplexFloat;

inline ComplexFloat c(float x) { return ComplexFloat(x); }
inline ComplexFloat c(float x, float y) { return ComplexFloat(x, y); }

inline ComplexFloat sin(ComplexFloat z) { return z.sin(); }
inline ComplexFloat cos(ComplexFloat z) { return z.cos(); }
inline ComplexFloat tan(ComplexFloat z) { return z.tan(); }
inline ComplexFloat asin(ComplexFloat z) { return z.asin(); }
inline ComplexFloat acos(ComplexFloat z) { return z.acos(); }
inline ComplexFloat atan(ComplexFloat z) { return z.atan(); }
inline ComplexFloat log(ComplexFloat z) { return z.log(); }
inline ComplexFloat exp(ComplexFloat z) { return c(E).pow(z); }
inline ComplexFloat sqrt(ComplexFloat z) { return z.sqrt(); }
inline ComplexFloat asinh(ComplexFloat z) { return z.asinh(); }
inline ComplexFloat sinh(ComplexFloat z) { return z.sinh(); }
inline ComplexFloat cosh(ComplexFloat z) { return z.cosh(); }
inline ComplexFloat pow(ComplexFloat x, ComplexFloat y) { return x.pow(y); }

// the complex func %%FUNC%%
// sed s/%%FUNC%%/"z*z"/g dc.metal > dcz.metal
ComplexFloat z_func(ComplexFloat z) { return %%FUNC%%;}


color HSV2int(float h, const float s, const float v);
color pixelColorzFunc(uint x, uint y, uint w, uint h);

inline color argbf2uint(uint8_t alpha, float r, float g, float b);
inline float pow2(float x) { return x*x; }
inline float pow3(float x) { return x*x*x; }
inline uint pos2index(uint2 position, uint width) { return position.x + width * position.y; }
inline uint geo2index(Geo geo) { return geo.x + geo.y * geo.width; }

kernel void domain_color( // pixel wise generation with z compiled func
                             device color*colors[[buffer(0)]], // buffer per device
                             device const uint*_geo[[buffer(1)]], // [w,h]

                             uint2 position [[thread_position_in_grid]] ) // 0..w*h
{
    uint w=_geo[0], h=_geo[1];

    colors[ pos2index(position, w) ] = pixelColorzFunc(position.x, position.y, w, h);
}

color pixelColorzFunc(uint x, uint y, uint w, uint h) {

    const float E = 2.7182818284590452353602874713527;
    const float M_PI = 3.141592653589793238462643383;
    const float PI = M_PI, PI2 = PI * 2.;
    const float limit=PI,  rmi = -limit, rma = limit, imi = -limit, ima = limit;

    auto z = ComplexFloat( ima - (ima - imi) * y / (h - 1),  rma - (rma - rmi) * x / (w - 1) );

    auto v = z_func(z); // evaluate zCode func

    auto ang = v.arg();
    while (ang < 0) ang += PI2;
    ang /= PI2;

    float m = v.abs(), ranges = 0., rangee = 1.; //  prop. e^n < m < e^(n-1)
    while (m > rangee) {
        ranges = rangee;
        rangee *= E;
    }
    float k = (m - ranges) / (rangee - ranges);
    float kk = (k < 0.5 ? k * 2. : 1. - (k - 0.5) * 2);

    float sat = 0.4 + (1. - pow3(1. - (kk)))     * 0.6;
    float val = 0.6 + (1. - pow3(1. - (1 - kk))) * 0.4;

    return HSV2int(ang, sat, val);
}


color HSV2int(float h, const float s, const float v) { // convert hsv to int with alpha 0xff00000
    float r = 0, g = 0, b = 0;

    if (s == 0) r = g = b = v;
    else {
        if (h == 1)  h = 0;

        float z = floor(h * 6.);
        int i = z;

        float   f = h * 6 - z,
        p = v * (1 - s), q = v * (1 - s * f),
        t = v * (1 - s * (1 - f));

        switch (i) {
            case 0:          r = v;   g = t;   b = p;             break;
            case 1:          r = q;   g = v;   b = p;             break;
            case 2:          r = p;   g = v;   b = t;             break;
            case 3:          r = p;   g = q;   b = v;             break;
            case 4:          r = t;   g = p;   b = v;             break;
            case 5:          r = v;   g = p;   b = q;             break;
        }
    }
    return argbf2uint(0xff, r,g,b);
}

inline color argbf2uint(uint8_t alpha, float r, float g, float b) { // alpha 0xff
    return (((color)alpha) << 24) |
    ( ( (color)(255.*r) & 0xff ) |  ( ((color)(255.*g)&0xff)<<8 ) | ( ((color)(255.*b)&0xff)<<16 ) );
}
