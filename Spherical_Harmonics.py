import sympy as sym

 # sympy variables
x, y, z = sym.symbols('x y z')
p = 2*sym.sqrt(sym.pi)

def Yl0(l): # m = 0 spherical harmonics
    a = sum([sym.binomial(l, k) * sym.binomial (sym.Rational((l+k-1)/2), l) * z**k for k in range(l+1)])
    a *= sym.sqrt(2*l+1) * 2**l
    return sym.simplify(a/p)

def Clm(l, m, bC): # z dependend part for m > 0
    a = sum([sym.binomial(l, k+m) * sym.binomial( sym.Rational((l+k+m-1)/2), l) * sym.Rational(sym.factorial(k+m) / sym.factorial(k)) * z**k for k in range(l-m+1)])
    a *= sym.sqrt(2*l+1) * sym.sqrt(2) * 2**l * sym.sqrt( sym.Rational( sym.factorial(l-m) / sym.factorial(l+m)) )
    return sym.simplify(a*bC/p)

def Slm(l, m, bS): # z dependend part for m < 0
    a = sum([sym.binomial(l, k+m) * sym.binomial( sym.Rational((l+k+m-1)/2), l) * sym.Rational(sym.factorial(k+m) / sym.factorial(k)) * z**k for k in range(l-m+1)])
    a *= sym.sqrt(2*l+1) * sym.sqrt(2) * 2**l * sym.sqrt( sym.Rational( sym.factorial(l-m) / sym.factorial(l+m)) )
    return sym.simplify(a*bS/p)

def bCm(m): # x, y dependend part for m > 0 
    return sum([(-1)**j * sym.binomial(m, 2*j)   * x**(m-2*j)   * y**(2*j)   for j in range(int( m   /2)+1)])

def bSm(m): # x, y dependend part for m < 0 
    return sum([(-1)**j * sym.binomial(m, 2*j+1) * x**(m-2*j-1) * y**(2*j+1) for j in range(int((m-1)/2)+1)])

def spherical_harmonics(l, m): # real sperical harmonics as function of x, y, z
    if m==0:
        return Yl0(l)
    elif m > 0:
        return Clm(l, m, bCm(m))
    else:
        return Slm(l, -m, bSm(-m))


def CSlm(l, m, bC, bS): # z dependend part for m > 0 and m < 0
    a = sum([sym.binomial(l, k+m) * sym.binomial( sym.Rational((l+k+m-1)/2), l) * sym.Rational(sym.factorial(k+m) / sym.factorial(k)) * z**k for k in range(l-m+1)])
    a *= sym.sqrt(2*l+1) * sym.sqrt(2) * 2**l * sym.sqrt( sym.Rational( sym.factorial(l-m) / sym.factorial(l+m)) )
    return sym.simplify(a*bC/p), sym.simplify(a*bS/p)

def spherical_harmonics_list(L): # all spherical harmonics up to l = L
    bC = [bCm(m) for m in range(1, L+1)]
    bS = [bSm(m) for m in range(1, L+1)]
    Sph = []
    for l in range(L+1):
        Sph.append(Yl0(l))
        for m in range(1, l+1):
            C, S = CSlm(l, m, bC[m-1], bS[m-1])
            Sph.append(C)
            Sph.append(S)
    return Sph


def jax_func(expr): # convert sympy expression into jax function
  return sym.lambdify([x, y, z], expr, 'jax')