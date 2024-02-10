import numpy
import numpy.linalg
import matplotlib.pyplot as plt          # For plotting

smin = 2                                  # Set minimal and maximal matrix size
smax = 9
rrs = numpy.ones(smax-smin+1)            # Pre-allocate arrays for relative residual,
res = numpy.ones(smax-smin+1)            # relative error and
mes = numpy.ones(smax-smin+1)            # maximal error

for n in range(smin,smax+1):             # Loop over matrix sizes
    xe = numpy.zeros((n,1))                               # Exact solution
    xe[0] = 1   
    A = numpy.zeros((n,n))               # Allocate and fill A
    for i in range(1,n):               
        for j in range(1,n):
            A[i-1,j-1] = (-1)**(i+j) / (i + 2*j)
    r = A[:,[0]]                                  # Set right-hand side
    x = numpy.linalg.solve(A,r)                                 # Solve by LUP decomposition

    condition_number = numpy.linalg.cond(A,2)
    
    rrs[n-smin] = numpy.linalg.norm(numpy.linalg.dot(A,x) - r, 2)/numpy.linalg.norm(r, 2)                          # Store relative residual, error and maximal error
    res[n-smin] = numpy.linalg.norm(x - xe,2) / numpy.linalg.norm(xe,2) 
    mes[n-smin] = condition_number * rrs[n-smin]


plt.semilogy(range(smin,smax+1),res,'-b',range(smin,smax+1),mes,'-r')
plt.xlim([smin,smax])
plt.xlabel('matrix size')
plt.ylabel('(maximal) relative error')
plt.title('Maximal (red) and actual (blue) relative error')
plt.show()
