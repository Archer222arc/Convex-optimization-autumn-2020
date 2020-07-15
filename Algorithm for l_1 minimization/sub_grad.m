function output = sub_grad(A,b,mu,opt)
% A,b,mu
% solve 0.5*norm(A*x-b)^2+mu*norm(x,1)
tic;
[m,n] = size(A);
if ~isfield(opt,'subsolver');   opt.subsolver = 'step_fixed';   end
if ~isfield(opt,'ite_max');     opt.ite_max = 1000; end
if ~isfield(opt,'x');           opt.x = randn(n,1); end
if ~isfield(opt,'tol');         opt.tol = 1e-4;     end

solver = opt.subsolver;
ite_max = opt.ite_max;
tol = opt.tol;
x = opt.x;
f = @(x) 0.5*norm(A*x-b)^2+mu*norm(x,1);
y = [];
x = {opt.x};
k = 0;
switch solver
%%
    case 'step_fixed'
        L = svds(A,1)^2+mu;
        t = 1/L;      
        while k < ite_max
            k = k+1;
            y = [y,f(x{k})];
            if k > 1
                res = (y(k)-y(k-1))/y(k-1);
                if abs(res) < tol;  break;  end
            end
            grad = A'*(A*x{k}-b)+mu*sign(x{k});
            x{k+1} = x{k}-t*grad;
        end
%%
end
output.time = toc;
output.x = x;
output.y = y;
output.k = k;