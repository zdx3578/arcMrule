%% ------------------------------------------------
%% bk.pl – background knowledge
%% ------------------------------------------------
%% Helper relations
greater_than(A,B) :- number(A), number(B), A > B.
equal(A,A).

%% ------------------------------------------------
%% Object facts
%% !!! AUTO‑GENERATED !!!
%% You can regenerate this section using extractor + a helper script.
%% Example stub:
%%
%% object(o1). color(o1, blue). holes(o1,2).
%% object(o2). color(o2, red).  holes(o2,0).