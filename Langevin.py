import numpy as np

def Langevin():
    """
    """

function Langevin_SDE

Nt = 100;
dt = 1;

N = 10;

g = 1;
s2 = 1;

f = zeros(N,Nt);
for ii = 1:N
    rng('shuffle')
    f(ii,:) = 0 + sqrt(dt*s2)*randn(Nt,1); % trajectory of noise 
end

rng('shuffle');
X0 = 0 + sqrt(s2/(2*g))*randn(N,1); % initial conditions

X = X0(1);   

F = @(t,X) (-g) * X;
G = @(t,X) diag(X) * diag(sqrt(s2));

SDE = sde(F, G, 'StartState', X)

[paths,times] = simulate(SDE,Nt, 'DeltaTime', dt);

end
