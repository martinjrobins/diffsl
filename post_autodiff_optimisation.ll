; ModuleID = '$name'
source_filename = "$name"

@enzyme_const_constants = hidden global [1 x double] zeroinitializer
@int_format_name = hidden global [10 x i8] c"name: %d\0A\00"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define void @set_u0(ptr noalias nocapture writeonly %0, ptr noalias nocapture readnone %1, i32 %2, i32 %3) local_unnamed_addr #0 {
entry:
  store double 1.000000e+00, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @calc_stop(double %0, ptr noalias nocapture readnone %1, ptr noalias nocapture readnone %2, ptr noalias nocapture readnone %3, i32 %4, i32 %5) local_unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @rhs(double %0, ptr noalias nocapture readonly %1, ptr noalias nocapture readnone %2, ptr noalias nocapture writeonly %3, i32 %4, i32 %5) local_unnamed_addr #2 {
entry:
  %F-02 = load double, ptr %1, align 8
  store double %F-02, ptr %3, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @mass(double %0, ptr noalias nocapture readnone %1, ptr noalias nocapture readnone %2, ptr noalias nocapture readnone %3, i32 %4, i32 %5) local_unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @calc_out(double %0, ptr noalias nocapture readonly %1, ptr noalias nocapture readnone %2, ptr nocapture writeonly %3, i32 %4, i32 %5) local_unnamed_addr #2 {
entry:
  %out-02 = load double, ptr %1, align 8
  store double %out-02, ptr %3, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @set_id(ptr nocapture readnone %0) local_unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define void @get_dims(ptr nocapture writeonly %0, ptr nocapture writeonly %1, ptr nocapture writeonly %2, ptr nocapture writeonly %3, ptr nocapture writeonly %4, ptr nocapture writeonly %5) local_unnamed_addr #0 {
entry:
  store i32 1, ptr %0, align 4
  store i32 0, ptr %1, align 4
  store i32 1, ptr %2, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %5, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @set_inputs(ptr nocapture readnone %0, ptr nocapture readnone %1) local_unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @get_inputs(ptr nocapture readnone %0, ptr nocapture readnone %1) local_unnamed_addr #1 {
entry:
  ret void
}

define void @set_constants(i32 %0, i32 %1) local_unnamed_addr {
entry:
  tail call void (ptr, ...) @printf(ptr nonnull @int_format_name, i32 0)
  store double 0x3FE0ACD00FE63B97, ptr @enzyme_const_constants, align 8
  ret void
}

declare extern_weak void @printf(ptr, ...) local_unnamed_addr

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
define double @sinh(double %x) local_unnamed_addr #3 {
entry:
  %sinh = fneg double %x
  %sinh1 = tail call double @llvm.exp.f64(double %sinh)
  %sinh2 = tail call double @llvm.exp.f64(double %x)
  %sinh3 = fsub double %sinh2, %sinh1
  %sinh5 = fmul double %sinh3, 5.000000e-01
  ret double %sinh5
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define void @get_constant_r(ptr nocapture writeonly %0, ptr nocapture writeonly %1) local_unnamed_addr #0 {
entry:
  store ptr @enzyme_const_constants, ptr %0, align 8
  store i32 1, ptr %1, align 4
  ret void
}

define void @set_u0_grad(ptr noalias %0, ptr noalias %1, ptr %2, ptr %3, i32 %4, i32 %5) {
entry:
  call void @fwddiffeset_u0(ptr %0, ptr %1, ptr %2, ptr %3, i32 %4, i32 %5)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @fwddiffeset_u0(ptr noalias nocapture writeonly %0, ptr nocapture %"'", ptr noalias nocapture readnone %1, ptr nocapture readnone %"'1", i32 %2, i32 %3) local_unnamed_addr #5 {
entry:
  store double 0.000000e+00, ptr %"'", align 8, !alias.scope !0, !noalias !3
  ret void
}

define void @rhs_grad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8) {
entry:
  call void @fwddifferhs(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @fwddifferhs(double %0, ptr noalias nocapture readonly %1, ptr nocapture %"'", ptr noalias nocapture readnone %2, ptr nocapture readnone %"'1", ptr noalias nocapture writeonly %3, ptr nocapture %"'2", i32 %4, i32 %5) local_unnamed_addr #5 {
entry:
  %"F-02'ipl" = load double, ptr %"'", align 8, !alias.scope !5, !noalias !8
  store double %"F-02'ipl", ptr %"'2", align 8, !alias.scope !10, !noalias !13
  ret void
}

define void @calc_out_grad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8) {
entry:
  call void @fwddiffecalc_out(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @fwddiffecalc_out(double %0, ptr noalias nocapture readonly %1, ptr nocapture %"'", ptr noalias nocapture readnone %2, ptr nocapture readnone %"'1", ptr nocapture writeonly %3, ptr nocapture %"'2", i32 %4, i32 %5) local_unnamed_addr #5 {
entry:
  %"out-02'ipl" = load double, ptr %"'", align 8, !alias.scope !15, !noalias !18
  store double %"out-02'ipl", ptr %"'2", align 8, !alias.scope !20, !noalias !23
  ret void
}

define void @set_inputs_grad(ptr noalias %0, ptr noalias %1, ptr %2, ptr %3) {
entry:
  call void @fwddiffeset_inputs(ptr %0, ptr %1, ptr %2, ptr %3)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @fwddiffeset_inputs(ptr nocapture readnone %0, ptr nocapture readnone %"'", ptr nocapture readnone %1, ptr nocapture readnone %"'1") local_unnamed_addr #5 {
entry:
  ret void
}

define void @set_u0_rgrad(ptr noalias %0, ptr noalias %1, ptr %2, ptr %3, i32 %4, i32 %5) {
entry:
  call void @diffeset_u0(ptr %0, ptr %1, ptr %2, ptr %3, i32 %4, i32 %5)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @diffeset_u0(ptr noalias nocapture writeonly %0, ptr nocapture %"'", ptr noalias nocapture readnone %1, ptr nocapture readnone %"'1", i32 %2, i32 %3) local_unnamed_addr #5 {
entry:
  store double 0.000000e+00, ptr %"'", align 8, !alias.scope !25, !noalias !28
  ret void
}

define void @rhs_rgrad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8) {
entry:
  call void @differhs(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @differhs(double %0, ptr noalias nocapture readonly %1, ptr nocapture %"'", ptr noalias nocapture readnone %2, ptr nocapture readnone %"'1", ptr noalias nocapture writeonly %3, ptr nocapture %"'2", i32 %4, i32 %5) local_unnamed_addr #5 {
entry:
  %6 = load double, ptr %"'2", align 8, !alias.scope !30, !noalias !33
  store double 0.000000e+00, ptr %"'2", align 8, !alias.scope !30, !noalias !33
  %7 = load double, ptr %"'", align 8, !alias.scope !35, !noalias !38
  %8 = fadd fast double %7, %6
  store double %8, ptr %"'", align 8, !alias.scope !35, !noalias !38
  ret void
}

define void @calc_out_rgrad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8) {
entry:
  call void @diffecalc_out(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @diffecalc_out(double %0, ptr noalias nocapture readonly %1, ptr nocapture %"'", ptr noalias nocapture readnone %2, ptr nocapture readnone %"'1", ptr nocapture writeonly %3, ptr nocapture %"'2", i32 %4, i32 %5) local_unnamed_addr #5 {
entry:
  %6 = load double, ptr %"'2", align 8, !alias.scope !40, !noalias !43
  store double 0.000000e+00, ptr %"'2", align 8, !alias.scope !40, !noalias !43
  %7 = load double, ptr %"'", align 8, !alias.scope !45, !noalias !48
  %8 = fadd fast double %7, %6
  store double %8, ptr %"'", align 8, !alias.scope !45, !noalias !48
  ret void
}

define void @set_inputs_rgrad(ptr noalias %0, ptr noalias %1, ptr %2, ptr %3) {
entry:
  call void @diffeset_inputs(ptr %0, ptr %1, ptr %2, ptr %3)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @diffeset_inputs(ptr nocapture readnone %0, ptr nocapture readnone %"'", ptr nocapture readnone %1, ptr nocapture readnone %"'1") local_unnamed_addr #5 {
entry:
  ret void
}

define void @mass_rgrad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8) {
entry:
  call void @diffemass(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define internal void @diffemass(double %0, ptr noalias nocapture readnone %1, ptr nocapture readnone %"'", ptr noalias nocapture readnone %2, ptr nocapture readnone %"'1", ptr noalias nocapture readnone %3, ptr nocapture readnone %"'2", i32 %4, i32 %5) local_unnamed_addr #5 {
entry:
  ret void
}

define void @rhs_full(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, i32 %4, i32 %5) {
entry:
  %y = getelementptr inbounds double, ptr %1, i32 0
  %thread_dim = alloca i32, align 4
  %thread_id = alloca i32, align 4
  %t = alloca double, align 8
  store double %0, ptr %t, align 8
  store i32 %4, ptr %thread_id, align 4
  store i32 %5, ptr %thread_dim, align 4
  br label %F-0

F-0:                                              ; preds = %F-0, %entry
  %i0 = phi i32 [ 0, %entry ], [ %F-05, %F-0 ]
  %F-01 = getelementptr inbounds double, ptr %y, i32 0
  %F-02 = load double, ptr %F-01, align 8
  %expr_index = mul i32 %i0, 1
  %acc = add i32 0, %expr_index
  %F-03 = add i32 0, %acc
  %F-04 = getelementptr inbounds double, ptr %3, i32 %F-03
  store double %F-02, ptr %F-04, align 8
  %F-05 = add i32 %i0, 1
  %F-06 = icmp ult i32 %F-05, 1
  br i1 %F-06, label %F-0, label %F-07

F-07:                                             ; preds = %F-0
  ret void
}

define void @rhs_full_grad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, i32 %6, i32 %7) {
entry:
  call void @fwddifferhs_full(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, i32 %6, i32 %7)
  ret void
}

; Function Attrs: mustprogress
define internal void @fwddifferhs_full(double %0, ptr noalias %1, ptr noalias %2, ptr %"'", ptr noalias %3, ptr %"'1", i32 %4, i32 %5) #6 {
entry:
  br label %F-0

F-0:                                              ; preds = %F-0, %entry
  %iv = phi i64 [ %iv.next, %F-0 ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %6 = trunc i64 %iv to i32
  %"F-04'ipg" = getelementptr inbounds double, ptr %"'1", i32 %6
  store double 0.000000e+00, ptr %"F-04'ipg", align 8, !alias.scope !50, !noalias !53
  %F-05 = add i32 %6, 1
  %F-06 = icmp ult i32 %F-05, 1
  br i1 %F-06, label %F-0, label %F-07

F-07:                                             ; preds = %F-0
  ret void
}

define void @rhs_full_rgrad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, i32 %6, i32 %7) {
entry:
  call void @differhs_full(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, i32 %6, i32 %7)
  ret void
}

; Function Attrs: mustprogress
define internal void @differhs_full(double %0, ptr noalias %1, ptr noalias %2, ptr %"'", ptr noalias %3, ptr %"'1", i32 %4, i32 %5) #6 {
entry:
  br label %F-0

F-0:                                              ; preds = %F-0, %entry
  %iv = phi i64 [ %iv.next, %F-0 ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %6 = trunc i64 %iv to i32
  %"F-04'ipg" = getelementptr inbounds double, ptr %"'1", i32 %6
  %F-05 = add i32 %6, 1
  %F-06 = icmp ult i32 %F-05, 1
  br i1 %F-06, label %F-0, label %invertF-0

invertentry:                                      ; preds = %invertF-0
  ret void

invertF-0:                                        ; preds = %F-0, %incinvertF-0
  %"iv'ac.0" = phi i64 [ %9, %incinvertF-0 ], [ 0, %F-0 ]
  %_unwrap = trunc i64 %"iv'ac.0" to i32
  %"F-04'ipg_unwrap" = getelementptr inbounds double, ptr %"'1", i32 %_unwrap
  store double 0.000000e+00, ptr %"F-04'ipg_unwrap", align 8, !alias.scope !55, !noalias !58
  %7 = icmp eq i64 %"iv'ac.0", 0
  %8 = xor i1 %7, true
  br i1 %7, label %invertentry, label %incinvertF-0

incinvertF-0:                                     ; preds = %invertF-0
  %9 = add nsw i64 %"iv'ac.0", -1
  br label %invertF-0
}

define void @calc_out_full(double %0, ptr noalias %1, ptr noalias %2, ptr %3, i32 %4, i32 %5) {
entry:
  %y = getelementptr inbounds double, ptr %1, i32 0
  %thread_dim = alloca i32, align 4
  %thread_id = alloca i32, align 4
  %t = alloca double, align 8
  store double %0, ptr %t, align 8
  store i32 %4, ptr %thread_id, align 4
  store i32 %5, ptr %thread_dim, align 4
  br label %out-0

out-0:                                            ; preds = %out-0, %entry
  %i0 = phi i32 [ 0, %entry ], [ %out-05, %out-0 ]
  %out-01 = getelementptr inbounds double, ptr %y, i32 0
  %out-02 = load double, ptr %out-01, align 8
  %expr_index = mul i32 %i0, 1
  %acc = add i32 0, %expr_index
  %out-03 = add i32 0, %acc
  %out-04 = getelementptr inbounds double, ptr %3, i32 %out-03
  store double %out-02, ptr %out-04, align 8
  %out-05 = add i32 %i0, 1
  %out-06 = icmp ult i32 %out-05, 1
  br i1 %out-06, label %out-0, label %out-07

out-07:                                           ; preds = %out-0
  ret void
}

define void @calc_out_full_grad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, i32 %6, i32 %7) {
entry:
  call void @fwddiffecalc_out_full(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, i32 %6, i32 %7)
  ret void
}

; Function Attrs: mustprogress
define internal void @fwddiffecalc_out_full(double %0, ptr noalias %1, ptr noalias %2, ptr %"'", ptr %3, ptr %"'1", i32 %4, i32 %5) #6 {
entry:
  br label %out-0

out-0:                                            ; preds = %out-0, %entry
  %iv = phi i64 [ %iv.next, %out-0 ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %6 = trunc i64 %iv to i32
  %"out-04'ipg" = getelementptr inbounds double, ptr %"'1", i32 %6
  store double 0.000000e+00, ptr %"out-04'ipg", align 8, !alias.scope !60, !noalias !63
  %out-05 = add i32 %6, 1
  %out-06 = icmp ult i32 %out-05, 1
  br i1 %out-06, label %out-0, label %out-07

out-07:                                           ; preds = %out-0
  ret void
}

define void @calc_out_full_rgrad(double %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr %4, ptr %5, i32 %6, i32 %7) {
entry:
  call void @diffecalc_out_full(double %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, i32 %6, i32 %7)
  ret void
}

; Function Attrs: mustprogress
define internal void @diffecalc_out_full(double %0, ptr noalias %1, ptr noalias %2, ptr %"'", ptr %3, ptr %"'1", i32 %4, i32 %5) #6 {
entry:
  br label %out-0

out-0:                                            ; preds = %out-0, %entry
  %iv = phi i64 [ %iv.next, %out-0 ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %6 = trunc i64 %iv to i32
  %"out-04'ipg" = getelementptr inbounds double, ptr %"'1", i32 %6
  %out-05 = add i32 %6, 1
  %out-06 = icmp ult i32 %out-05, 1
  br i1 %out-06, label %out-0, label %invertout-0

invertentry:                                      ; preds = %invertout-0
  ret void

invertout-0:                                      ; preds = %out-0, %incinvertout-0
  %"iv'ac.0" = phi i64 [ %9, %incinvertout-0 ], [ 0, %out-0 ]
  %_unwrap = trunc i64 %"iv'ac.0" to i32
  %"out-04'ipg_unwrap" = getelementptr inbounds double, ptr %"'1", i32 %_unwrap
  store double 0.000000e+00, ptr %"out-04'ipg_unwrap", align 8, !alias.scope !65, !noalias !68
  %7 = icmp eq i64 %"iv'ac.0", 0
  %8 = xor i1 %7, true
  br i1 %7, label %invertentry, label %incinvertout-0

incinvertout-0:                                   ; preds = %invertout-0
  %9 = add nsw i64 %"iv'ac.0", -1
  br label %invertout-0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { mustprogress nofree norecurse nosync nounwind }
attributes #6 = { mustprogress }

!0 = !{!1}
!1 = distinct !{!1, !2, !"shadow_0"}
!2 = distinct !{!2, !" diff: %"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"primal"}
!5 = !{!6}
!6 = distinct !{!6, !7, !"shadow_0"}
!7 = distinct !{!7, !" diff: %"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"primal"}
!10 = !{!11}
!11 = distinct !{!11, !12, !"shadow_0"}
!12 = distinct !{!12, !" diff: %"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"primal"}
!15 = !{!16}
!16 = distinct !{!16, !17, !"shadow_0"}
!17 = distinct !{!17, !" diff: %"}
!18 = !{!19}
!19 = distinct !{!19, !17, !"primal"}
!20 = !{!21}
!21 = distinct !{!21, !22, !"shadow_0"}
!22 = distinct !{!22, !" diff: %"}
!23 = !{!24}
!24 = distinct !{!24, !22, !"primal"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"shadow_0"}
!27 = distinct !{!27, !" diff: %"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"primal"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"shadow_0"}
!32 = distinct !{!32, !" diff: %"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"primal"}
!35 = !{!36}
!36 = distinct !{!36, !37, !"shadow_0"}
!37 = distinct !{!37, !" diff: %"}
!38 = !{!39}
!39 = distinct !{!39, !37, !"primal"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"shadow_0"}
!42 = distinct !{!42, !" diff: %"}
!43 = !{!44}
!44 = distinct !{!44, !42, !"primal"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"shadow_0"}
!47 = distinct !{!47, !" diff: %"}
!48 = !{!49}
!49 = distinct !{!49, !47, !"primal"}
!50 = !{!51}
!51 = distinct !{!51, !52, !"shadow_0"}
!52 = distinct !{!52, !" diff: %"}
!53 = !{!54}
!54 = distinct !{!54, !52, !"primal"}
!55 = !{!56}
!56 = distinct !{!56, !57, !"shadow_0"}
!57 = distinct !{!57, !" diff: %"}
!58 = !{!59}
!59 = distinct !{!59, !57, !"primal"}
!60 = !{!61}
!61 = distinct !{!61, !62, !"shadow_0"}
!62 = distinct !{!62, !" diff: %"}
!63 = !{!64}
!64 = distinct !{!64, !62, !"primal"}
!65 = !{!66}
!66 = distinct !{!66, !67, !"shadow_0"}
!67 = distinct !{!67, !" diff: %"}
!68 = !{!69}
!69 = distinct !{!69, !67, !"primal"}
