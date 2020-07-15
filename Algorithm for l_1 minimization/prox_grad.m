function output = prox_grad(A,b,mu,opt)
% A,b,mu
% solve 0.5*norm(A*x-b)^2+mu*norm(x,1)
[m,n] = size(A);
if ~isfield(opt,'subsolver');      opt.subsolver = 'KL';     end
if ~isfield(opt,'ite_max');     opt.ite_max = 1000; end
if ~isfield(opt,'x');           opt.x = randn(n,1); end
if ~isfield(opt,'tol');         opt.tol = 1e-4;     end
if ~isfield(opt,'warm');        opt.warm = 'no'; tic ;         % warm up
else; if ~isfield(opt,'ite');    opt.ite = 0;    tic; end;  end
solver = opt.subsolver;
ite_max = opt.ite_max;
tol = opt.tol;
x = opt.x;
f = @(x) 0.5*norm(A*x-b)^2;
g = @(x) mu*norm(x,1);
F = @(x) f(x)+g(x);
y = [];
x = {opt.x};
switch solver
%%    
    case 'linesearch_simple'
        t = 1;
        k = 0;
        eta = 0.5;
        while k < ite_max
            k = k+1;
            y = [y,F(x{k})];
            if k > 1
               res = (y(k)-y(k-1))/y(k-1);
               if abs(res) < tol;    break;  end
            end
            % line search
            grad = A'*(A*x{k}-b);
            x0 = x-t*grad;
            while F(x0) > F(x)
                t = t*eta;
                x0 = x{k}-t*grad;
            end
            x0 = sign(x0).*max(abs(x0)-mu*t,0);
            x{k+1} = x0;
        end
%%
    case 'KL'
        k = 0;
        while k < ite_max
            k = k+1;
            y = [y,F(x{k})];
            if k > 1
                res = (y(k)-y(k-1))/y(k-1);
                if abs(res) < tol;   break;  end
            end
            % 调和
            t = 2e-2/k;
            grad = A'*(A*x{k}-b);
            x0 = x-t*grad;
            x0 = sign(x0).*max(abs(x0)-mu*t,0);
            x{k+1} = x0;
        end
%%
    case 'Inertial'
        k = 0;
        t = 1e-4;
        beta = 0.5;
        while k < ite_max
            k = k+1;
            y = [y,F(x{k})];
            if k > 1
                res = (y(k)-y(k-1))/y(k-1);
                if abs(res) < tol;  break;  end
            end
            grad = A'*(A*x{k}-b);
            if k > 1;   x0 = x{k}-t*grad+beta*(x{k}-x{k-1}); 
            else x0 = x{k}-t*grad;  end
            x{k+1} = sign(x0).*max(abs(x0)-mu*t,0);
        end
    case 'BB'
        k = 0;
        eta = 0.9;
        beta = 0.8;
        t = 0.5;
        while k < ite_max
            k = k+1;
            y = [y,F(x{k})];
            if k > 1
                res = (y(k)-y(k-1))/y(k-1);
                if abs(res) < tol;  break;  end
                grad = A'*(A*x{k}-b);
                s = x{k}-x{k-1};    z = A'*(A*(x{k}-x{k-1}));
                tau = s'*z/(z'*z);
                x0 = x{k}-tau*grad;
                x0 = sign(x0).*max(abs(x0)-tau*mu,0);
                d = x0-x{k};
                Lap = grad'*d+g(x0)-g(x{k});
                while F(x{k}+t*d) > F(x{k})+beta*t*Lap
                    t = t*eta;
                end
                x{k+1} = x{k}+t*d;
            else
                grad = A'*(A*x{1}-b);
                t = 1;
                while F(x{1}-t*grad)>y(1)
                    t = t*eta;
                end
                x0 = x{1}-t*grad;
                x{2} = sign(x0).*max(abs(x0)-mu*t,0);
            end
        end
%%
end
if strcmpi(opt.warm,'yes')
    if norm(opt.x-x{k}) < tol
        if opt.ite == 0;   output.time = toc;   end   
        output.x = x;
        output.y = y;
        output.k = k;
        output.restart = 1;
    else   
        opt_sub = opt;
        opt_sub.x = x{k};
        opt_sub.ite = 1;
        output_sub = prox_grad(A,b,mu,opt_sub);
        output.x = [x,output_sub.x];
        output.y = [y,output_sub.y];
        output.restart = output_sub.restart+1;
        output.k = output_sub.k+k;
        if opt.ite == 0;   output.time = toc;    end 
    end
else
    output.time = toc; 
    output.x = x;
    output.y = y;
    output.k = k;
end