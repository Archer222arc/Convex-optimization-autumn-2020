function output = usepack(A,b,mu,opt)
% 调用 mosek 和 gurobi
[m,n] = size(A);
if strcmpi(opt.pack,'mosek')
    H = A'*A;
    H = [H,zeros(n);zeros(n),zeros(n)];
    f = [-A*b;ones(n,1)];
    A = [speye(n);-speye(n)];
    b = 