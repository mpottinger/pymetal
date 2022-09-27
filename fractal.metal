//
// fractal.metal
//
// xcrun -sdk macosx metal -c fractal.metal -o fractal.air
// xcrun -sdk macosx metallib fractal.air -o fractal.metallib

#include <metal_stdlib>
using namespace metal;

template<typename T>
struct Complex
{
    T re=0, im=0;

    inline Complex() { re = im = 0; }
    inline Complex(T re, T im) : re(re), im(im) { }
    inline Complex(T re) : re(re), im(0) { }

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

constant const float E = 2.7182818284590452353602874713527;
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

constant int n_palette= 256;
constant int fire_palette[n_palette]={0, 0, 4, 12, 16, 24, 32, 36, 44, 48, 56, 64, 68, 76, 80, 88, 96, 100, 108, 116, 120, 128, 132,
                         140, 148, 152, 160, 164, 172, 180, 184, 192, 200, 1224, 3272, 4300, 6348, 7376, 9424, 10448,
                         12500, 14548, 15576, 17624, 18648, 20700, 21724, 23776, 25824, 26848, 28900, 29924, 31976,
                         33000, 35048, 36076, 38124, 40176, 41200, 43248, 44276, 46324, 47352, 49400, 51452, 313596,
                         837884, 1363196, 1887484, 2412796, 2937084, 3461372, 3986684, 4510972, 5036284, 5560572,
                         6084860, 6610172, 7134460, 7659772, 8184060, 8708348, 9233660, 9757948, 10283260, 10807548,
                         11331836, 11857148, 12381436, 12906748, 13431036, 13955324, 14480636, 15004924, 15530236,
                         16054524, 16579836, 16317692, 16055548, 15793404, 15269116, 15006972, 14744828, 14220540,
                         13958396, 13696252, 13171964, 12909820, 12647676, 12123388, 11861244, 11599100, 11074812,
                         10812668, 10550524, 10288380, 9764092, 9501948, 9239804, 8715516, 8453372, 8191228, 7666940,
                         7404796, 7142652, 6618364, 6356220, 6094076, 5569788, 5307644, 5045500, 4783356, 4259068,
                         3996924, 3734780, 3210492, 2948348, 2686204, 2161916, 1899772, 1637628, 1113340, 851196,
                         589052, 64764, 63740, 62716, 61692, 59644, 58620, 57596, 55548, 54524, 53500, 51452, 50428,
                         49404, 47356, 46332, 45308, 43260, 42236, 41212, 40188, 38140, 37116, 36092, 34044, 33020,
                         31996, 29948, 28924, 27900, 25852, 24828, 23804, 21756, 20732, 19708, 18684, 16636, 15612,
                         14588, 12540, 11516, 10492, 8444, 7420, 6396, 4348, 3324, 2300, 252, 248, 244, 240, 236, 232,
                         228, 224, 220, 216, 212, 208, 204, 200, 196, 192, 188, 184, 180, 176, 172, 168, 164, 160, 156,
                         152, 148, 144, 140, 136, 132, 128, 124, 120, 116, 112, 108, 104, 100, 96, 92, 88, 84, 80, 76,
                         72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 0, 0};

//constant ComplexFloat center=c(0.5, 0), // could be parameters
//                      range=c(-2, 1.7);

typedef uint32_t color; // aa bb gg rr  32 bit color

inline uint pos2index(uint2 position, uint width) { return position.x + width * position.y; }
ComplexFloat do_scale(ComplexFloat cr, ComplexFloat range, int i, int j, int w, int h) {
            return cr + ComplexFloat((range.im - range.re) * i / w,
                                     (range.im - range.re) * j / h);
}

color mandelbrot(uint i, uint j, uint w, uint h, ComplexFloat center, ComplexFloat range) {
        float scale = 0.8, ratio = w / h;
        int iter = 200, ix;
        ComplexFloat cr = c(range.re, range.re),
                     c0 = ComplexFloat(c(scale * ratio) * do_scale(cr, range, i, j, w, h) - center),
                     z = c0;

        for (ix=0; ix<iter; ix++) {
             z = z * z + c0;  // z*z is the typical 2nd order fractal
             if (z.abs() > 2) break;
        }
        return 0xff000000 | ( (ix == iter - 1) ? 0 : fire_palette[(n_palette * ix / 50) % n_palette] );
}

kernel void fractal(  device color*colors             [[buffer(0)]], // output color image
                      device const uint2&size         [[buffer(1)]], // (w,h) as a numpy([w,h], dtype=np.int32)
                      device const ComplexFloat&center[[buffer(2)]],
                      device const ComplexFloat&range [[buffer(3)]],

                      uint2 position [[thread_position_in_grid]] ) // 0..w*h
{
    colors[ pos2index(position, size.x) ] = mandelbrot(position.x, position.y, size.x, size.y, center, range);
}

