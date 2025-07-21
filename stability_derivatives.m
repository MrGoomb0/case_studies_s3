% A320
% C_Lift from .txt files
% alpha_A320 = [-0.8, -0.4, -0.2, -0.1, 0, 0.2, 0.23, 0.26, 0.29, 0.31, 0.4, 0.8];
% C_L_A320 = [-1.185, -0.987, -0.698, -0.489, 0.038, 1.154, 1.279, 1.39, 1.249, 1.159, 1.031, 1.207];
% 
% alpha_A320 = [0, 0.2, 0.23, 0.26, 0.29, 0.31, 0.4, 0.8];
% C_L_A320 = [0.038, 1.154, 1.279, 1.39, 1.249, 1.159, 1.031, 1.207];


figure(1);
% subplot(2, 3, 1)
% % Obvious abrupt stall for C_L starting from alpha_star = 0.26
% 
% C_L_fit_part_1_A320 = fit(alpha_A320(1:4)', C_L_A320(1:4)', 'poly2');
% hold on 
% plot((-0.8:0.05:0.26), C_L_fit_part_1_A320((-0.8:0.05:0.26)), '--r', 'Linewidth',2);
% C_L_fit_part_2_A320 = fit(alpha_A320(4:end)', C_L_A320(4:end)', 'poly2');
% plot((0.26:0.01:0.8), C_L_fit_part_2_A320((0.26:0.01:0.8)), '--r', 'Linewidth',2);
% 
% scatter(alpha_A320, C_L_A320, 'filled', 'red')
% legend("", "C_L fit", "C_L data", "location", "northwest")
% 
% title("C_L(\alpha) graph for A320")
% ylabel("C_L")
% xlabel("\alpha")
% 
% % C_Drag by using C_Drag = C_D0 + 1 / (4 * e * AR) * C_L^2
% 
% AR_A320 = 44^2 / 1450; % Wingspan^2 / Wingarea
% e_A320 = 0.9;
% C_D0_A320 = 0.0227;
% C_D_A320 = C_D0_A320 + 1 / (pi * e_A320 * AR_A320) .* C_L_A320.^2;
% ylim([0, 2])
% xlim([-0.02,1])
% 
% subplot(2, 3, 2)
% 
% % C_D_fit_part_1_A320 = fit(alpha_A320(1:5)', C_D_A320(1:5)', 'poly2');
% % hold on 
% % plot((-0.8:0.05:0), C_D_fit_part_1_A320((-0.8:0.05:0)), '--r', 'Linewidth',2);
% % C_D_fit_part_2_A320 = fit(alpha_A320(5:8)', C_D_A320(5:8)', 'poly1');
% % plot((0:0.01:0.26), C_D_fit_part_2_A320((0:0.01:0.26)), '--r', 'Linewidth',2);
% % C_D_fit_part_3_A320 = fit(alpha_A320(8:end)', C_D_A320(8:end)', 'poly2');
% % plot((0.26:0.01:0.8), C_D_fit_part_3_A320((0.26:0.01:0.8)), '--r', 'Linewidth',2);
% 
% C_D_fit_part_1_A320 = C_D0_A320 + 1 / (pi * e_A320 * AR_A320) * C_L_fit_part_1_A320((-0.8:0.01:0.26)).^2;
% hold on 
% plot((-0.8:0.01:0.26), C_D_fit_part_1_A320, ':b', 'Linewidth',2);
% C_D_fit_part_2_A320 = C_D0_A320 + 1 / (pi * e_A320 * AR_A320) * C_L_fit_part_2_A320(0.26:0.01:0.8).^2;
% plot((0.26:0.01:0.8), C_D_fit_part_2_A320, ':b', 'Linewidth',2);
% 
% scatter(alpha_A320, C_D_A320, 'filled', 'red')
% legend("", "", "C_D fit", "", "C_D calculated","C_D data", "location", "northwest")
% ylim([0, 0.8])
% xlim([-0.02,1])
% 
% title("C_D(\alpha) graph for A320")
% ylabel("C_D")
% xlabel("\alpha")
% 
% subplot(2, 3, 3)
% scatter(C_D_A320, C_L_A320, 'filled')
% title("C_L i.f.o. C_D graph for A320")
% ylabel("C_L")
% xlabel("C_D")


% A320neo
% C_Lift

% alpha_A320neo = [-0.5, -0.2, 0, 0.044, 0.087, 0.139, 0.2, 0.215, 0.23, 0.26, 0.29, 0.31, 0.4, 0.8];
% C_L_A320neo = [-1.389, -0.87, 0.165, 0.380, 0.620, 0.885, 1.12, 1.58, 1.79, 1.875, 1.84, 1.81, 1.68, 0.7925];

syms alpha

alpha_A320neo = [0, 0.044, 0.087, 0.139, 0.2, 0.215, 0.23, 0.26, 0.29, 0.31, 0.4];
C_L_A320neo = [0.165, 0.380, 0.620, 0.885, 1.12, 1.58, 1.79, 1.875, 1.84, 1.81, 1.68];


subplot(1, 3, 1)

x0 = [0]; 
endpoint = [0.26, 1.875];
fitfun = fittype( @(c_alpha,x) c_alpha*x + (endpoint(2) - c_alpha*endpoint(1)));

% C_L_fit_part_1_A320neo = fit(alpha_A320neo(1:10)', C_L_A320neo(1:10)', 'poly1')
C_L_fit_part_1_A320neo = fit(alpha_A320neo(1:8)', C_L_A320neo(1:8)',fitfun,'StartPoint',x0);
c_alpha_A_1 = C_L_fit_part_1_A320neo.c_alpha
c_alpha_A_0 = endpoint(2) - c_alpha_A_1 * endpoint(1)


hold on 
plot((-0.5:0.01:0.26), C_L_fit_part_1_A320neo((-0.5:0.01:0.26)), '--r', 'Linewidth',2);

x0 = [1, 1]; 
endpoint = [0.26, 1.875];
fitfun = fittype( @(c_alpha_1,c_alpha_2,x) c_alpha_2*x.^2 + + c_alpha_1*x + (endpoint(2) - c_alpha_2*endpoint(1).^2 - c_alpha_1*endpoint(1)));

C_L_fit_part_2_A320neo = fit(alpha_A320neo(8:end)', C_L_A320neo(8:end)',fitfun,'StartPoint',x0);
c_alpha_B_1 = C_L_fit_part_2_A320neo.c_alpha_1
c_alpha_B_2 = C_L_fit_part_2_A320neo.c_alpha_2
c_alpha_B_0 = endpoint(2) - endpoint(1)^2 * c_alpha_B_2 - endpoint(1) * c_alpha_B_1


plot((0.26:0.01:0.8), C_L_fit_part_2_A320neo((0.26:0.01:0.8)), '--r', 'Linewidth',2);
ylim([0, 2])
xlim([-0.02,0.5])

scatter(alpha_A320neo, C_L_A320neo, 'filled', 'red')
legend("", "C_L fit", "C_L data", "location", "northeast")
title("C_L(\alpha) graph for A320neo")
ylabel("C_L")
xlabel("\alpha")

C_L_part_1 = @(alpha) c_alpha_A_0 + c_alpha_A_1 * alpha;
C_L_part_2 = @(alpha) c_alpha_B_0 + c_alpha_B_1 * alpha + c_alpha_B_2 * alpha.^2;
C_L = @(alpha) piecewise((alpha < 0.26), C_L_part_1(alpha), (0.26 < alpha), C_L_part_2(alpha));

% C_Drag
AR_A320neo = 117.454^2 / 1313.2; % Wingspan^2 / Wingarea
e_A320neo = 0.75;
C_D0_A320neo = 0.0237;
C_D_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) .* C_L_A320neo.^2;

subplot(1, 3, 2)

% C_D_fit_part_1_A320neo = fit(alpha_A320neo(1:3)', C_D_A320neo(1:3)', 'poly1');
% hold on 
% plot((-0.5:0.05:0), C_D_fit_part_1_A320neo((-0.5:0.05:0)), '--r', 'Linewidth',2);
% C_D_fit_part_2_A320neo = fit(alpha_A320neo(3:10)', C_D_A320neo(3:10)', 'poly2');
% plot((0:0.01:0.26), C_D_fit_part_2_A320neo((0:0.01:0.26)), '--r', 'Linewidth',2);
% C_D_fit_part_3_A320neo = fit(alpha_A320neo(10:end)', C_D_A320neo(10:end)', 'poly2');
% plot((0.26:0.01:0.8), C_D_fit_part_3_A320neo((0.26:0.01:0.8)), '--r', 'Linewidth', 2);

C_D_fit_part_1_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_fit_part_1_A320neo((-0.5:0.01:0.26)).^2;
hold on 
plot((-0.5:0.01:0.26), C_D_fit_part_1_A320neo, '--red', 'Linewidth',2);
C_D_fit_part_2_A320neo = C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_fit_part_2_A320neo(0.26:0.01:0.8).^2;
plot((0.26:0.01:0.8), C_D_fit_part_2_A320neo, '--red', 'Linewidth',2);

C_D_part_1 = @(alpha) C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_1(alpha).^2;
% C_D_part_2 = vpa(collect(C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_2.^2));
% coefficients = coeffs(C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_2.^2);
% C_D_part_2_truncated = @(alpha) coefficients(1) + alpha*coefficients(2) + alpha.^2*coefficients(3);

x0 = [1, 1];
endpoint = [0.26, C_D_part_1(0.26)]
fitfun = fittype( @(c_alpha_1,c_alpha_2,x) c_alpha_2*x.^2 + + c_alpha_1*x + (endpoint(2) - c_alpha_2*endpoint(1).^2 - c_alpha_1*endpoint(1)));

C_D_part_2 = fit(alpha_A320neo(8:end)', C_D_A320neo(8:end)',fitfun,'StartPoint',x0);

plot((0.26:0.01:0.8), C_D_part_2(0.26:0.01:0.8), ':blue', 'Linewidth',2);


scatter(alpha_A320neo, C_D_A320neo, 'filled', 'red')

legend("", "C_D calculated", "C_D fitted","C_D data", "location", "northeast")
ylim([0.02, 0.2])
xlim([-0.02,0.5])

title("C_D(\alpha) graph for A320neo")
ylabel("C_D")
xlabel("\alpha")


subplot(1, 3, 3)
scatter(C_D_A320neo, C_L_A320neo, 'filled')
title("C_L i.f.o. C_D graph for A320neo")
ylabel("C_L")
xlabel("C_D")

N = 60;
% T_0 = (-14115.48307 + N*1076.26550) * 1/0.303 * 1/0.454
T_0 = 130 * 10^3 / (0.303*0.454);
T_1 = -22.58;

figure(2)
p1 = plot(C_D_A320neo, C_L_A320neo, '-ored', 'linewidth', 2);
p1.MarkerEdgeColor = 'red';
p1.MarkerFaceColor = 'red';

ylabel("C_L")
xlabel("C_D")
xlim([0, 0.62])
ylim([0, 3.5])

lgd = legend("A320neo", "Location", 'northwest')

fontsize(lgd,70,'points')
        fontsize(70,"points")

set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on

figure(3)
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
% title("C_L i.f.o. C_D graph for B-727")
ylabel("C_L")
xlabel("C_D")
xlim([0, 0.62])
ylim([0, 3.5])

lgd = legend("B-727", "Location", 'northwest')

fontsize(lgd,70,'points')
        fontsize(70,"points")
set(gcf,'Units','inches');
        screenposition = get(gcf,'Position');
        set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize',[screenposition(3:4)]);
        axis square;
        box on

N = 60;
% T_0 = (-14115.48307 + N*1076.26550) * 1/0.303 * 1/0.454
T_0 = 130 * 10^3 / (0.303*0.454);
T_1 = -22.58;
alpha_star = 0.26;
C_0 = 0.0423; C_1 = 7.0489; C_2 = 2.0920; C_3 = -0.4711; C_4 = -1.3978;
C_L_part_1 = (C_0 + C_1 .* alpha);
C_L_part_2 = (C_2 + C_3 .* alpha + C_4 .* alpha.^2);
C_D_part_1 = vpa(C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_1.^2)
C_D_part_2 = vpa(C_D0_A320neo + 1 / (pi * e_A320neo * AR_A320neo) * C_L_part_2.^2)

vpa(collect(C_D_part_1))
vpa(collect(C_D_part_2))

% 
% syms alpha
% C_L = @(alpha) -(-0.4256 - 7.3903 .* alpha + 0.22845 .* alpha.^2);
% C_D = @(alpha) -0.4256 - 4.0662 .* alpha - 3.18165 .* alpha.^2;
% alpha_grid = -0.5:0.01:0.8;
% 
% figure(2)
% subplot(2, 1, 1)
% plot(alpha_grid, C_L(alpha_grid));
% subplot(2, 1, 2)
% plot(alpha_grid, C_D(alpha_grid));
% 
% syms alpha
% C_L = @(alpha) -(-0.4256 - 7.3903 .* alpha + 0.49.* alpha.^2);
% C_D = @(alpha) -0.4256 - 4.0662 .* alpha - 3.18165 .* alpha.^2;
% alpha_grid = -0.5:0.01:0.8;
% 
% figure(3)
% subplot(2, 1, 1)
% plot(alpha_grid, C_L(alpha_grid));
% subplot(2, 1, 2)
% plot(alpha_grid, C_D(alpha_grid));