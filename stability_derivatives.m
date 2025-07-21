% This files makes it possible to approximate the needed coefficients for
% the A320neo model.

% C_Lift

syms alpha

alpha_A320neo = [0, 0.044, 0.087, 0.139, 0.2, 0.215, 0.23, 0.26, 0.29, 0.31, 0.4];
C_L_A320neo = [0.165, 0.380, 0.620, 0.885, 1.12, 1.58, 1.79, 1.875, 1.84, 1.81, 1.68];

figure(1)

% Fit of the first part (alpha <= 0.26)
x0 = [0]; 
endpoint = [0.26, 1.875];
fitfun = fittype( @(c_alpha,x) c_alpha*x + (endpoint(2) - c_alpha*endpoint(1)));

C_L_fit_part_1_A320neo = fit(alpha_A320neo(1:8)', C_L_A320neo(1:8)',fitfun,'StartPoint',x0);
c_alpha_A_1 = C_L_fit_part_1_A320neo.c_alpha;
c_alpha_A_0 = endpoint(2) - c_alpha_A_1 * endpoint(1);

plot((-0.5:0.01:0.26), C_L_fit_part_1_A320neo((-0.5:0.01:0.26)), '--k', 'Linewidth',2);

% Fit of the second part (alpha <= 0.26)
x0 = [1, 1]; 
endpoint = [0.26, 1.875];
fitfun = fittype( @(c_alpha_1,c_alpha_2,x) c_alpha_2*x.^2 + + c_alpha_1*x + (endpoint(2) - c_alpha_2*endpoint(1).^2 - c_alpha_1*endpoint(1)));

C_L_fit_part_2_A320neo = fit(alpha_A320neo(8:end)', C_L_A320neo(8:end)',fitfun,'StartPoint',x0);
c_alpha_B_1 = C_L_fit_part_2_A320neo.c_alpha_1;
c_alpha_B_2 = C_L_fit_part_2_A320neo.c_alpha_2;
c_alpha_B_0 = endpoint(2) - endpoint(1)^2 * c_alpha_B_2 - endpoint(1) * c_alpha_B_1;

hold on
plot((0.26:0.01:0.8), C_L_fit_part_2_A320neo((0.26:0.01:0.8)), '--k', 'Linewidth',2);
ylim([0, 2])
xlim([-0.02,0.5])

scatter(alpha_A320neo, C_L_A320neo, 'filled', 'red');
lgd = legend("", "C_L fit", "C_L data", "location", "northwest");
ylabel("C_L")
xlabel("\alpha")

C_L_part_1 = @(alpha) c_alpha_A_0 + c_alpha_A_1 * alpha;
C_L_part_2 = @(alpha) c_alpha_B_0 + c_alpha_B_1 * alpha + c_alpha_B_2 * alpha.^2;
C_L = @(alpha) piecewise((alpha < 0.26), C_L_part_1(alpha), (0.26 < alpha), C_L_part_2(alpha));

fontsize(lgd,70,'points')
        fontsize(70,"points")
set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on


% C_Drag

AR_A320neo = 117.454^2 / 1313.2; % Wingspan^2 / Wingarea
e_A320neo = 0.75;
C_D0_A320neo = 0.0237;
C_D_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) .* C_L_A320neo.^2;

figure(2)

C_D_fit_part_1_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_fit_part_1_A320neo((-0.5:0.01:0.26)).^2;
hold on 
plot((-0.5:0.01:0.26), C_D_fit_part_1_A320neo, '--k', 'Linewidth',2);
C_D_fit_part_2_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_fit_part_2_A320neo(0.26:0.01:0.8).^2;
plot((0.26:0.01:0.8), C_D_fit_part_2_A320neo, '--k', 'Linewidth',2);

C_D_part_1 = @(alpha) C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_1(alpha).^2;

x0 = [1, 1];
endpoint = [0.26, C_D_part_1(0.26)]
fitfun = fittype( @(c_alpha_1,c_alpha_2,x) c_alpha_2*x.^2 + + c_alpha_1*x + (endpoint(2) - c_alpha_2*endpoint(1).^2 - c_alpha_1*endpoint(1)));

C_D_part_2 = fit(alpha_A320neo(8:end)', C_D_A320neo(8:end)',fitfun,'StartPoint',x0);

plot((0.26:0.01:0.8), C_D_part_2(0.26:0.01:0.8), ':blue', 'Linewidth',2);

scatter(alpha_A320neo, C_D_A320neo, 'filled', 'red')

lgd = legend("", "C_D calculated", "C_D fitted","C_D data", "location", "northeast");
ylim([0.02, 0.2])
xlim([-0.02,0.5])

ylabel("C_D")
xlabel("\alpha")

fontsize(lgd,70,'points')
        fontsize(70,"points")
set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on

% Extra figures

figure(3)
p1 = plot(C_D_A320neo, C_L_A320neo, '-ored', 'linewidth', 2);
p1.MarkerEdgeColor = 'red';
p1.MarkerFaceColor = 'red';

ylabel("C_L")
xlabel("C_D")
xlim([0, 0.62])
ylim([0, 3.5])

lgd = legend("A320neo", "Location", 'northwest');

fontsize(lgd,70,'points')
        fontsize(70,"points")

set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on

figure(5)

B_0 = 0.1552; B_1 = 0.12368; B_2 = 2.4203; C_0 = 0.7125; C_1 = 6.0877; C_2 = -9.0277; alpha_star = 17.2 / 180 * pi;
C_D_original = B_0 + B_1*alpha_A320neo + B_2 * alpha_A320neo.^2;
C_L_original_part_1 = C_0 + C_1 * alpha_A320neo;
C_L_original_part_2 = C_0 + C_1 * alpha_A320neo + C_2 * (alpha_A320neo- alpha_star).^2;

p2 = plot(C_D_original(1:8), C_L_original_part_1(1:8), '-oblue', 'linewidth',2);
p2.MarkerEdgeColor = "blue";
p2.MarkerFaceColor = "blue";
hold on
p2 = plot(C_D_original(8:end), C_L_original_part_2(8:end), '-oblue', 'linewidth',2);
p2.MarkerEdgeColor = "blue";
p2.MarkerFaceColor = "blue";

ylabel("C_L")
xlabel("C_D")
xlim([0, 0.62])
ylim([0, 3.5])

lgd = legend("B-727", "Location", 'northwest');

fontsize(lgd,70,'points')
        fontsize(70,"points")
set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on