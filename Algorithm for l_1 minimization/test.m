% Lasso
% minimize norm(A*x-b)^2/2+mu*norm(x,1)
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
n = 1024;
m = 512;
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;
mu = 1e-3;
F = @(x)0.5*sum_square(A*x-b)+mu*norm(x,1);
solver = 'sub_grad';

opt.tol = 1e-8;
opt.subsolver = 1;
opt.ite_max = 10000;
% x = prox_grad(A,b,mu,opt);

switch solver
    case 'cvx_mosek'
        cvx_begin
            variable x(n)
            cvx_solver Mosek
            minimize(F(x))
        cvx_end
    case 'cvx_gurobi'
        cvx_begin
            variable x(n)
            cvx_solver Gurobi
            minimize(F(x))
        cvx_end
%     case 'mosek'
%         
    case 'prox_grad'
        opt.subsolver = 'Inertial';
        opt.warm = 'no';
        output = prox_grad(A,b,mu,opt);       
        x = output.x;
        y = output.y;
        k = output.k;
        semilogy(1:k,y);
    case 'sub_grad'
        opt.subsolver = 'step_fixed';
        output = sub_grad(A,b,mu,opt);
        x = output.x;
        y = output.y;
        k = output.k;
        semilogy(1:k,y);
end