%% Generate report figures from the real prototype evaluation outputs.
% Run from MATLAB with the project folder as Current Folder:
%   run('matlab/generate_report_figures.m')
% This script does not create or substitute synthetic results. It stops if
% the Python training run has not produced the required real-data CSV files.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
predFile = fullfile(projectRoot, 'reports', 'test_predictions.csv');
lossFile = fullfile(projectRoot, 'reports', 'training_history.csv');
outDir = fullfile(projectRoot, 'reports', 'matlab_figures');

if ~isfile(predFile) || ~isfile(lossFile)
    error(['Required real-evaluation CSVs were not found. First prepare the reviewed MRI manifest and run ' ...
        'scripts/train_multimodal.py. Expected: reports/test_predictions.csv and reports/training_history.csv.']);
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

P = readtable(predFile, 'TextType', 'string');
H = readtable(lossFile);
requiredP = ["actual_label", "predicted_probability", "predicted_label", "sex"];
requiredH = ["epoch", "train_loss"];
if ~all(ismember(requiredP, string(P.Properties.VariableNames)))
    error('test_predictions.csv is missing one or more required columns.');
end
if ~all(ismember(requiredH, string(H.Properties.VariableNames)))
    error('training_history.csv is missing one or more required columns.');
end

y = double(P.actual_label(:));
score = double(P.predicted_probability(:));
yhat = double(P.predicted_label(:));
sex = lower(strtrim(string(P.sex(:))));
if isempty(y) || any(~ismember(y, [0 1])) || any(~isfinite(score)) || any(score < 0 | score > 1)
    error('Evaluation CSV has empty or invalid labels/probabilities.');
end
if numel(unique(y)) < 2
    error('Overall ROC requires both outcome classes in the held-out set.');
end
if any(~ismember(yhat, [0 1]))
    error('Predicted labels must contain only 0 and 1.');
end

%% Figure 1: training loss, explicitly not validation loss.
f = figure('Color','w','Position',[100 100 760 520]);
plot(H.epoch, H.train_loss, '-o', 'LineWidth', 1.8, 'MarkerSize', 4, 'Color',[0.12 0.35 0.62]);
grid on; box off;
xlabel('Epoch'); ylabel('Mean training binary cross-entropy');
title('Training loss (training set only)');
savePng(f, fullfile(outDir,'figure1_training_loss.png'));
close(f);

%% Figure 2: held-out receiver operating characteristic curve.
[fpr, tpr, auc] = rocPoints(y, score);
f = figure('Color','w','Position',[100 100 700 560]);
plot(fpr, tpr, 'LineWidth', 2.2, 'Color',[0.08 0.39 0.68]); hold on;
plot([0 1],[0 1],'--','Color',[0.55 0.55 0.55],'LineWidth',1.2);
axis([0 1 0 1]); axis square; grid on; box off;
xlabel('False-positive rate (1 - specificity)'); ylabel('True-positive rate (sensitivity)');
title(sprintf('Held-out ROC curve (AUC = %.3f, n = %d)', auc, numel(y)));
savePng(f, fullfile(outDir,'figure2_heldout_roc.png'));
close(f);

%% Figure 3: confusion matrix at the prototype's pre-specified 0.50 threshold.
C = confusionCounts(y, yhat);
f = figure('Color','w','Position',[100 100 650 540]);
imagesc(C); axis image; colormap(f, [0.91 0.94 0.97; 0.12 0.31 0.53]);
colorbar; caxis([0 max(1,max(C(:)))]);
set(gca,'XTick',1:2,'XTickLabel',{'sMCI (0)','pMCI (1)'}, ...
    'YTick',1:2,'YTickLabel',{'sMCI (0)','pMCI (1)'},'FontSize',10);
xlabel('Predicted class'); ylabel('Actual class');
title(sprintf('Held-out confusion matrix (threshold = 0.50, n = %d)',numel(y)));
for r = 1:2
    for c = 1:2
        text(c,r,sprintf('%d',C(r,c)),'HorizontalAlignment','center', ...
            'FontSize',16,'FontWeight','bold','Color',[0.05 0.08 0.12]);
    end
end
savePng(f, fullfile(outDir,'figure3_confusion_matrix.png'));
close(f);

%% Figure 4: subgroup performance, with counts shown for context.
groups = ["female", "male"];
metricNames = {'AUC','Sensitivity','Specificity','F1-score','False-negative rate'};
M = nan(numel(groups),numel(metricNames));
counts = zeros(numel(groups),1);
for g = 1:numel(groups)
    idx = (sex == groups(g));
    counts(g) = sum(idx);
    if counts(g) == 0, continue; end
    M(g,1) = safeAuc(y(idx), score(idx));
    [~,~,~,~,m] = binaryMetrics(y(idx), yhat(idx));
    M(g,2:5) = m;
end
f = figure('Color','w','Position',[100 100 820 540]);
bar(M','grouped'); grid on; box off;
ylim([0 1]); ylabel('Score');
set(gca,'XTick',1:numel(metricNames),'XTickLabel',metricNames,'FontSize',10);
legend({sprintf('Female (n=%d)',counts(1)),sprintf('Male (n=%d)',counts(2))}, ...
    'Location','southoutside','Orientation','horizontal');
title('Exploratory held-out performance by sex');
text(0.01,-0.20,'NaN bars indicate a metric could not be calculated for that subgroup.', ...
    'Units','normalized','FontSize',9,'Color',[0.35 0.35 0.35]);
savePng(f, fullfile(outDir,'figure4_sex_stratified_metrics.png'));
close(f);

%% Captions file for direct use in the report.
[~,overallSensitivity,overallSpecificity] = binaryMetrics(y,yhat);
captions = { ...
    'Fig. 1. Training loss across epochs for the training partition of the exploratory model. This is not a validation curve and does not establish generalization.'; ...
    sprintf('Fig. 2. Receiver operating characteristic curve on the participant-level held-out test partition (n = %d; AUC = %.3f). Estimates are exploratory and should be interpreted with the cohort size and split strategy in mind.',numel(y),auc); ...
    sprintf('Fig. 3. Confusion matrix for the held-out test partition using the prototype threshold of 0.50 (n = %d; sensitivity = %.3f; specificity = %.3f). Rows denote actual outcomes and columns denote predictions.',numel(y),overallSensitivity,overallSpecificity); ...
    'Fig. 4. AUC, sensitivity, specificity, F1-score, and false-negative rate for female and male participants in the same held-out test partition; subgroup counts appear in the legend. These small-sample subgroup estimates are descriptive, not evidence of a sex difference.' ...
    };
fid = fopen(fullfile(outDir,'figure_captions.txt'),'w');
cleanupObj = onCleanup(@() fclose(fid));
for k = 1:numel(captions), fprintf(fid,'%s\n\n',captions{k}); end
fprintf('Saved four report figures and captions to: %s\n',outDir);

%% Local functions (base MATLAB only).
function savePng(fig,path)
    set(fig,'InvertHardcopy','off');
    print(fig,path,'-dpng','-r300');
end

function [fpr,tpr,auc] = rocPoints(y,s)
    [s,order] = sort(s(:),'descend'); y = y(order);
    P = sum(y==1); N = sum(y==0);
    if P==0 || N==0, error('ROC curve needs both classes.'); end
    change = [find(diff(s)~=0); numel(s)];
    tp = cumsum(y==1); fp = cumsum(y==0);
    tpr = [0; tp(change)/P]; fpr = [0; fp(change)/N];
    auc = trapz(fpr,tpr);
end

function a = safeAuc(y,s)
    if numel(unique(y))<2, a=NaN; return; end
    [~,~,a] = rocPoints(y,s);
end

function C = confusionCounts(y,yhat)
    C = zeros(2,2);
    for k=1:numel(y)
        C(y(k)+1,yhat(k)+1) = C(y(k)+1,yhat(k)+1)+1;
    end
end

function [C,sens,spec,f1,m] = binaryMetrics(y,yhat)
    C = confusionCounts(y,yhat);
    tn=C(1,1); fp=C(1,2); fn=C(2,1); tp=C(2,2);
    sens=tp/max(1,tp+fn); spec=tn/max(1,tn+fp);
    precision=tp/max(1,tp+fp); f1=2*precision*sens/max(eps,precision+sens);
    fnr=fn/max(1,fn+tp);
    m=[sens spec f1 fnr];
end
