%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ARMAX(0,0,1)-GARCH(1,1) model parameter estimation using
% Markov Chain Monte Carlo Method (Metropolis-Hastings)
%
% Author: Changwei Xiong, 06/09/2011
% 
% ARMAX(0,0,1)-GARCH(1,1) process described as below
% 
% x, y are two time series:     y(t) = a + b * x(t) + u(t)
% e is an residual series:      u(t) ~ Normal(0, h(t))
% variance for residual series: h(t) = c + d * u(t-1)^2 + e * h(t-1)
% 
% 	a: intercept of linear regression
% 	b: slope of linear regression
% 	c: conditional mean variance
% 	d: coefficients related to lagged-1 residuals
% 	e: coefficient related to lagged-1 conditional variances
%
% In this model, the parameters to be estimated are 
% the "a", "b", "c", "d" and "e".
%
%(C) Copyright 2011, Changwei Xiong (axcw@hotmail.com)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MCMC_ARMAX_GARCH()
    clear all
    close all
    clc

    % length of series
    N = 2000;
    
    % original parameters
    % y(t) = a + b*x(t) + u(t)
    % u(t) = sqrt(h(t)) * N(0,1)
    % h(t) = c + d*u(t-1)^2 + e*h(t-1);
    a = 0.2; % intercept of linear regression
    b = 0.9; % slope of linear regression
    c = 0.2; % conditional mean variance
    d = 0.3; % coefficients related to lagged-1 residuals
    e = 0.4; % coefficient related to lagged-1 conditional variances
    
    % show the original parameters
    disp '------------------------------------------'
    disp 'Original parameters:'
    Parameters = [a b c d e]   

    % generate a GARCH(1,1)-Regression series
    [x, y, spec] = Gen_ARMAX_GARCH(N, a, b, c, d, e);    
    
    % use the built-in garchfit function to estimate the parameters
    % this is based on MLE method, and will be used as a benchmark
    [estimates, stderrors] = garchfit(spec, y, x);
    disp '------------------------------------------'
    disp 'Estimated by built-in "garchfit" function:'
    
    Garchfit_Estimates = [ ...
        estimates.C, ...
        estimates.Regress, ...
        estimates.K, ...
        estimates.ARCH, ...
        estimates.GARCH]
    
    Garchfit_Standard_Errors = [ ...
        stderrors.C,  ...
        stderrors.Regress, ...
        stderrors.K,  ...
        stderrors.ARCH,  ...
        stderrors.GARCH]
    
    % total # of MCMC simulations 
    Q = 100000;
    % 2-D array that stores MCMC estimations
    est2d = ones(Q,5) * 0.5;
    
    % assume proposal density is normal and use the following sigma's to 
    % generate the samples
    sigma = [0.013 0.013 0.025 0.03 0.045];
    
    for i=2:Q
        est = est2d(i-1, :);
        for j = 1:5
            % generate new sample and progress
            new = normrnd(est(j), sigma(j));
            est = MetropolisHastings(new, j, est, x, y);
        end
        est2d(i, :) = est;
    end
    
    % remove the burn-in period
    est2d = est2d(5000:end, :);
    disp '------------------------------------------'
    disp 'Estimated by MCMC method:'
    % calculate the estimation mean
    MCMC_est = mean(est2d, 1) 
    % calculate the estimation std. dev.
    MCMC_est_stderr = std(est2d, 1)
    disp '------------------------------------------'
    
    % plot estimates distribution
    figure
    titles = ['a', 'b', 'c', 'd', 'e'];
    for i = 1:5
        subplot(3,2,i)
        hist(est2d(:,i), 30);
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor',[.8 .8 1])
        title( ...
            [titles(i), ...
            '  ({\mu}=', ...
            num2str(MCMC_est(i), '%1.3f'), ...
            ', {\sigma}=', ...
            num2str(MCMC_est_stderr(i), '%.3f'), ...
            ')'], ...
            'fontsize',14);
    end
end



function [x, y, spec] = Gen_ARMAX_GARCH(N, a, b, c, d, e)
    x = randn(N, 1) * 10;
    spec = garchset('C',a, 'Regress',b, 'K',c, 'ARCH',d ,'GARCH',e, 'Display','off');
    [innovation, sigma, y] = garchsim(spec, N, 1, [], x);
end



function llh = loglikelihood(est, x, y)
% Calculate the loglikelihood, assuming it follows a normal distribution
% the constant term is ignored
    a = est(1);
    b = est(2);
    c = est(3);
    d = est(4);
    e = est(5);
    
    u2 = (y - a - b * x).^2;
    N = length(x);
    
    h = zeros(N, 1);
    for t = 1:N
        if (t == 1)
            h(t) = c;
        else
            h(t) = c + d * u2(t-1) + e * h(t-1);
        end
    end    
    llh = -0.5 * sum(log(h) + u2./h);
end



function est = MetropolisHastings(new, j, est, x, y)
% MH method, assumes prior distribution is flat, and the proposal density
% is symmetric (normal distribution). 
% this is basically a random-walk Metropolis Hastings method.
    op = est(:);% old parameters
    np = est(:);
    np(j) = new;% new parameters to be tested
    % parameters out of the constraints are treated with zero probability
    if (np(3)<0 || np(4)< 0 || np(5)< 0 || (np(4)+np(5))>1)
        est = op;
        return
    end
    oldllh = loglikelihood(op, x, y);
    newllh = loglikelihood(np, x, y);
 
    p = exp(newllh-oldllh);
    if(p > unifrnd(0,1))
        est = np;
    else
        est = op;
    end
end




