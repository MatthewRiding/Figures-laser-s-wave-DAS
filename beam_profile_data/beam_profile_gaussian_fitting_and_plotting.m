%% Load data:

% Line profiles extracted from phosphor image:
% Long axis:
load("profile_phosphor_long_axis.mat")
% Thin axis:
load("profile_phosphor_thin_axis.mat")

% 2026 versions:
% Long axis:
load("profile_phosphor_long_axis_2026.mat")
% Thin axis:
load("profile_phosphor_thin_axis_2026.mat")

% Line profiles extracted from IR image:
% Long axis:
load("profile_IR_long_axis.mat")
% Thin axis:
load("profile_IR_thin_axis.mat")

%% Fit gaussian functions to line profiles:

fig1 = fit_gaussian_and_plot( ...
    imagejlineprofiledataphosphorimagelongaxis, ...
    imagejlineprofiledataphosphorimagethinaxis, ...
    'gauss2', 'gauss1', ...
    0, 40, ...
    'Phosphor 2025');

fig2 = fit_gaussian_and_plot( ...
    imagej_length_profile_phosphor_long_axis_2026, ...
    imagej_width_profile_phosphor_short_axis_2026, ...
    'gauss2', 'gauss1', ...
    0, 40, ...
    'Phosphor 2026');

fig3 = fit_gaussian_and_plot( ...
    imagejlineprofiledataIRimagelongaxis, ...
    imagejlineprofiledataIRimagethinaxis, ...
    'gauss2', 'gauss1', ...
    0, 0, ...
    'IR Data');

%% Function definitions:

function [my_fit, FWHM] = fit_gaussian_beam_profile(x, intensity, model, background)
    my_fit = fit(x, intensity - background, model);
    FWHM = my_fit.c1 * 2 * sqrt(log(2));
end

function FWHM = plot_data_and_fit(table_data, model, background)
    x = table_data.X;
    intensity = table_data.Y;
    [my_fit, FWHM] = fit_gaussian_beam_profile(x, intensity, model, background);
    plot(my_fit, x, intensity-background)
end

function fig = fit_gaussian_and_plot( ...
    table_thin, table_long, ...
    model_thin, model_long, ...
    background_thin, background_long, ...
    sg_title_string)
    fig = figure(Position=[100 100 500, 400]);
        subplot(2, 1, 1);
            FWHM_thin = plot_data_and_fit(table_thin, model_thin, background_thin);
            title(sprintf(' FWHM = %.2f mm', FWHM_thin))
        subplot(2, 1, 2);
            FWHM_long = plot_data_and_fit(table_long, model_long, background_long);
            title(sprintf(' FWHM = %.2f mm', FWHM_long))
        sgtitle(sg_title_string)
end
