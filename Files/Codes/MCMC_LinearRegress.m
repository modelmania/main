%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Linear Regression model parameter estimation using
% Markov Chain Monte Carlo Method (Metropolis-Hastings)
%
% Author: Changwei Xiong, 06/09/2011
% 
% The Linear Regression model is in the form: 
% 
%    y(t) = a + b*x1(t) + c*x2(t) + e(t),   e(t) ~ Normal(0, s^2) 
% 
% 	a: intercept of linear regression
% 	b: slope of linear regression on x1
% 	c: slope of linear regression on x2
% 	s: constant standard deviation of residuals
%
% In this model, the parameters to be estimated are 
% the "a", "b", "c", and the "s".
%
%(C) Copyright 2011, Changwei Xiong (axcw@hotmail.com)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MCMC_LinearRegress()
    clear all
    close all
    clc

    % length of series
    N = 500;
    
	% original parameters
	% y = a + b*x1 + c*log(x1) + N(0,s)
    a = 1; % intercept
    b = 2; % slope for x1
    c = 3; % slope for x2
    s = 1.2; % std. dev. for error term 
	
    disp '------------------------------------------'
    disp 'Original parameters:'
	% show the original parameters
    Parameters = [a b c s]
    
    % Simulate linear regression series
	% y = a + b*x1 + c*log(x1) + N(0,s)
    x1 = sort(unifrnd(0, 10, N, 1));
    x2 = log(x1);
    y = a + b * x1 + c * x2 + normrnd(0, s, N, 1);    
    
    % use the built-in regress function to estimate the parameters
    % this is based on OLS/MLE method, and will be used as a benchmark	
    est = regress(y, [ones(size(x1)) x1 x2])';
    std_est = std(y - est(1) - est(2)*x1 - est(3)*x2);
    disp '------------------------------------------'
    disp 'Estimated by built-in "regress" function:'
    Regress_est = [est, std_est]
    
	% total # of MCMC simulations
    Q = 100000;
	% 2-D array that stores MCMC estimations
    est2d = ones(Q,4);
	% assume proposal density is normal and use the following sigma's to 
    % generate the samples
    sigma = [0.1 0.04 0.1 0.04];

    for i=2:Q
        est = est2d(i-1, :);
        for j = 1:4    
			% generate new sample and progress        
            new = normrnd(est(j), sigma(j));
            est = MetropolisHastings(new, j, est, x1, x2, y);
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
    titles = ['a', 'b', 'c', 's'];
    for i = 1:4
        subplot(2,2,i)
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


function f = loglikelihood(est, x1, x2, y)
% Calculate the loglikelihood, assuming it follows a normal distribution
% the constant term is ignored
    a = est(1);
    b = est(2);
    c = est(3);
    s = est(4);
    v = s*s;
    z = y - a - b*x1 - c*x2;
    f = -0.5 * sum(log(v) + z.^2/v);
end

function est = MetropolisHastings(new, j, est, x1, x2, y)
% MH method, assumes prior distribution is flat, and the proposal density
% is symmetric (normal distribution). 
% this is basically a random-walk Metropolis Hastings method.
    op = est(:); % old parameters
    np = est(:); 
    np(j) = new; % new parameters to be tested

    oldllh = loglikelihood(op, x1, x2, y);
    newllh = loglikelihood(np, x1, x2, y);
 
    p = exp(newllh-oldllh);
    if(p > unifrnd(0,1))
        est = np;
    else
        est = op;
    end
end



