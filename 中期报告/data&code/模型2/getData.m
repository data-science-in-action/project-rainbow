function [r,d] = getData()
    [data,~,~] = xlsread('C:\Users\lenovo\Desktop\wuhan.xls')

    data(isnan(data))=0;

    pat=data(:,1);
    pat_re=data(:,2);
    pat_die=data(:,3);

    [fitresult,~] = getRe(pat, pat_re);
    r=fitresult.p1;
    [fitresult, ~] = getDie(pat, pat_die);
    d=fitresult.p1;
end




