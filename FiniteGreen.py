import numpy as np

def FiniteGreen(g, n):
    """
    """

function GN = GN(g,n)


[g_n,t] = size(g);
%Choose a random number between 1 and 10.

numbas = rand(1,n);

GN = zeros(1,t);

for ii = 1:n
    sample = numbas(ii)*g_n;%Pick this sample of G(t).
    sample = round(sample);
    if(sample == 0)
        sample = g_n;
    end
    GN = GN + g(sample,:);
end

GN = GN/n; % average of n randomly selected trajectories of g



end
