clear all ; close all hidden; clc ;
load('pimaindiansdiabetes');
cump1=0;
mt1=0;
cump2=0;
mt2=0;
itr=10;
for k=1:10
for r=0:0.1:10
for l=1:itr;
    p1=0;p2=0;
    %% make Train And Test
    [TrainData,TrainTarget,TestData,TestTarget] =...
        MakeTestAndTrainData(pimaindiansdiabetes);
    
    % c1=0
    c1=TrainData(TrainTarget(:,1)==0,:);
    c1Target=TrainTarget(TrainTarget(:,1)==0,:);
    %c2=1
    c2=TrainData(TrainTarget(:,1)==1,:);
    c2Target=TrainTarget(TrainTarget(:,1)==1,:);
    
    %Permute TestData
    TestOut=TestTarget;
    TestIn=TestData;
    
    %% MCIS:
    % for haberman 1:0.7 , 2:0.6 , 3:0.6 , 4:0.5 , 5:0.4 , 6:0.5
    %              7:0.4 , 8:0.4 , 9:0.4 , 10:0.4
    % for pimaindiansdiabetes 1:1.4 , 2:1.2 , 3:1.2 , 4:1.1 , 5:1 , 6:1.1
    %                         7:1 , 8:1 , 9:1.1 , 10:1
    % for ionosphere 
    %     
    % for breastcancerwisconsin k=1 r>1.8 , k=2 r>1.9 , k=3 r>1.6
%                               k=4 r>1.7 , k=5 r>1.6 , k=6 r>1.6
%                               k=7 r>1.6 , k=8 r>1.7 , k=9 r>1.6
%                               k=10 r>1.6
    
%     k=1; %Number of cluster
%     r=.6; % Radios of nearest cluster
    
    [data1,data2,data1Target,data2Target]=MCIS(c1,c2,c1Target,c2Target,r,k);
    
    %% My Widrow_Hoff
    
    train_patterns=[data1;data2];
    train_targets=[data1Target;data2Target];
    test_patterns=TestIn;
    tic
    [test_targets]=Widrow_Hoff(train_patterns,train_targets,test_patterns);
    t1=toc;
    [c,cm,ind,per]=confusion(TestOut',test_targets');
    
    p1=1-c;
    cump1=cump1+p1;
    mt1=mt1+t1;
    
    %% Real Widrow_Hoff
    
    train_patterns=[c1;c2];
    train_targets=[c1Target;c2Target];
    test_patterns=TestIn;
    tic
    [Real_test_targets]=Widrow_Hoff(train_patterns,train_targets,test_patterns);
    t2=toc;
    [cn2,cm,ind,per]=confusion(TestOut',Real_test_targets');
    
    p2=1-cn2;
    cump2=cump2+p2;
    mt2=mt2+t2;
end

percent1=(cump1/itr)*100;
percent2=(cump2/itr)*100;

if(percent1-percent2>=0)
    disp([k r percent1 mt1/itr percent2 mt2/itr]);
end
percent1=0;mt1=0;cump1=0;
percent2=0;mt2=0;cump2=0;
end %end k
end %end r
clear ans mt1 mt2 t1 t2 c cm cn2 cump1 cump2 ind ...
    itr k l p1 p2 r percent1 percent2 per test_patterns ...
    TestIn TestOut train_patterns train_targets;