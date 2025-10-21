# Overfitting Detector Gains - CodeLlama Exp 2

Overfitting detector gain bugs: 4

These bugs had **no correct patches before OD refinement**. All pre-refinement patches were overfitting ??they passed tests but did not actually fix the root cause. OD refinement produced correct patches for these bugs.

---

## Jsoup-32

**Root cause:** Element.clone() performed a shallow copy of the lazily cached classNames set, so the clone could retain or derive a shared mutable Set with the original element.

**Why pre-patch overfits:** The pre-patch fixes the shared-mutable-set problem only when this.classNames is already non-null, but it is not generally correct because classNames is lazily initialized and may legitimately be null at clone time. On such reasonable inputs, clone.classNames = new HashSet<String>(this.classNames) will fail, so the patch overfits to the tested scenario where classNames() was called before cloning.

**OD refinement (NECESSARY):** The refinement added a null check before copying this.classNames into a new HashSet. This changes runtime behavior: the pre-patch throws a NullPointerException when clone() is called on an Element whose classNames cache has not yet been initialized, whereas the refined patch preserves the lazy-null state and avoids that failure.

**Verdict:** Because the refinement introduced necessary null-handling logic that the pre-patch lacked, the original patch was not semantically correct in general. It passed the existing tests only under the specific condition that classNames had already been initialized.

**Before OD (overfitting):**
```diff
@@ -1,5 +1,5 @@
 public Element clone() {
 Element clone = (Element) super.clone();
-clone.classNames();
+clone.classNames = new HashSet<String>(this.classNames);
 return clone;
 }
```

**After OD (correct):**
```diff
@@ -1,5 +1,7 @@
 public Element clone() {
 Element clone = (Element) super.clone();
-clone.classNames();
+if (this.classNames != null) {
+clone.classNames = new HashSet<String>(this.classNames);
+}
 return clone;
 }
```

**Ground truth fix:**
```diff
@@ -1,5 +1,5 @@
 public Element clone() {
 Element clone = (Element) super.clone();
-clone.classNames();
+clone.classNames = null; // derived on first hit, otherwise gets a pointer to source classnames
 return clone;
 }
```

---

## Lang-14

**Root cause:** The bug is that StringUtils.equals delegates to CharSequence.equals, but many CharSequence implementations do not define content-based equality across different CharSequence types.

**Why pre-patch overfits:** The pre-patch fixes the observed failing tests, but it does not correctly address the general root cause. For example, two distinct StringBuilder instances with the same characters have the same class, so the pre-patch calls StringBuilder.equals (identity), returning false instead of true; similarly, same-class custom CharSequence implementations without content-based equals would be mishandled.

**OD refinement (NECESSARY):** The refinement changed the special-case from "same concrete class" to "both are String" before using equals, and otherwise falls back to content comparison via toString(). This is necessary because the pre-patch incorrectly trusts equals for any two objects of the same CharSequence class, even though same-class CharSequence implementations may still use identity-based or otherwise non-content-based equals.

**Verdict:** Because the OD refinement introduced necessary logic to avoid relying on equals for arbitrary same-class CharSequence objects, the pre-patch was a true overfitting patch rather than an alternative correct fix.

**Before OD (overfitting):**
```diff
@@ -1,9 +1,6 @@
 public static boolean equals(CharSequence cs1, CharSequence cs2) {
-if (cs1 == cs2) {
-return true;
+if (cs1 == cs2) return true;
+if (cs1 == null || cs2 == null) return false;
+if (cs1.getClass().equals(cs2.getClass())) return cs1.equals(cs2);
+return cs1.toString().equals(cs2.toString());
 }
-if (cs1 == null || cs2 == null) {
-return false;
-}
-return cs1.equals(cs2);
-}
```

**After OD (correct):**
```diff
@@ -1,9 +1,6 @@
 public static boolean equals(CharSequence cs1, CharSequence cs2) {
-if (cs1 == cs2) {
-return true;
+if (cs1 == cs2) return true;
+if (cs1 == null || cs2 == null) return false;
+if (cs1 instanceof String && cs2 instanceof String) return ((String) cs1).equals((String) cs2);
+else return cs1.toString().equals(cs2.toString());
 }
-if (cs1 == null || cs2 == null) {
-return false;
-}
-return cs1.equals(cs2);
-}
```

**Ground truth fix:**
```diff
@@ -6,4 +6,7 @@
 return false;
 }
+if (cs1 instanceof String && cs2 instanceof String) {
 return cs1.equals(cs2);
 }
+return CharSequenceUtils.regionMatches(cs1, false, 0, cs2, 0, Math.max(cs1.length(), cs2.length()));
+}
```

---

## Math-53

**Root cause:** Complex.add fails to preserve the class invariant that any complex number with a NaN real or imaginary part must be represented as Complex.NaN, including when the left-hand operand itself is NaN.

**Why pre-patch overfits:** The pre-patch fixes the specific failing test scenario where the right-hand operand has a NaN component, but it does not fully fix the root cause. For example, `Complex.NaN.add(new Complex(1, 2))` would bypass the `rhs` check and return `createComplex(NaN, NaN)` rather than the canonical `Complex.NaN`, unlike the ground-truth behavior.

**OD refinement (NECESSARY):** The refinement added a post-addition check `if (result.isNaN()) return Complex.NaN;`, which covers NaN arising from either operand, not just from `rhs`. This is necessary because the pre-patch only inspects `rhs` and misses cases where `this` is already NaN, so it can return a non-canonical Complex with NaN components instead of the required `Complex.NaN`.

**Verdict:** Because the refinement introduced necessary missing logic to handle NaN results beyond the specific tested `rhs` case, the pre-patch was overfitting rather than a complete semantic fix.

**Before OD (overfitting):**
```diff
@@ -1,6 +1,5 @@
-public Complex add(Complex rhs)
-throws NullArgumentException {
+public Complex add(Complex rhs) throws NullArgumentException {
 MathUtils.checkNotNull(rhs);
-return createComplex(real + rhs.getReal(),
-imaginary + rhs.getImaginary());
+if (Double.isNaN(rhs.getReal()) || Double.isNaN(rhs.getImaginary())) return Complex.NaN;
+return createComplex(real + rhs.getReal(), imaginary + rhs.getImaginary());
 }
```

**After OD (correct):**
```diff
@@ -1,6 +1,11 @@
-public Complex add(Complex rhs)
-throws NullArgumentException {
+public Complex add(Complex rhs) throws NullArgumentException {
 MathUtils.checkNotNull(rhs);
-return createComplex(real + rhs.getReal(),
-imaginary + rhs.getImaginary());
+if (Double.isNaN(rhs.getReal()) || Double.isNaN(rhs.getImaginary())) {
+return Complex.NaN;
 }
+Complex result = createComplex(real + rhs.getReal(), imaginary + rhs.getImaginary());
+if (result.isNaN()) {
+return Complex.NaN;
+}
+return result;
+}
```

**Ground truth fix:**
```diff
@@ -2,4 +2,7 @@
 throws NullArgumentException {
 MathUtils.checkNotNull(rhs);
+if (isNaN || rhs.isNaN) {
+return NaN;
+}
 return createComplex(real + rhs.getReal(),
 imaginary + rhs.getImaginary());
```

---

## Mockito-12

**Root cause:** The bug is caused by blindly casting the first generic type argument to Class even when that argument is itself a ParameterizedType, which triggers a ClassCastException for nested generics.

**Why pre-patch overfits:** The pre-patch does address the reported nested-generic failure by extracting the raw type when the first type argument is a ParameterizedType, so it passes the given tests. However, it does not fully fix the root cause in a robust way because it still assumes every non-parameterized type argument is a Class, which would fail on reasonable generic declarations like ArgumentCaptor<T> or ArgumentCaptor<? extends Foo>.

**OD refinement (NECESSARY):** The refinement added a type check before casting non-parameterized type arguments: it now returns the Class only when the argument is actually a Class, and otherwise falls back to Object.class. This is semantically important because the pre-patch still performs an unsafe cast in the else branch for any Type that is neither ParameterizedType nor Class, such as TypeVariable or WildcardType.

**Verdict:** Because the OD refinement is NECESSARY and adds missing safety logic for non-Class Type arguments, the pre-patch is not semantically complete. It passes the current tests but remains overfitting due to unsafe casting on valid reflective Type cases outside the tested inputs.

**Before OD (overfitting):**
```diff
@@ -2,9 +2,13 @@
 Type generic = field.getGenericType();
 if (generic != null && generic instanceof ParameterizedType) {
-Type actual = ((ParameterizedType) generic).getActualTypeArguments()[0];
-return (Class) actual;
-//in case of nested generics we don't go deep
+Type[] typeArguments = ((ParameterizedType) generic).getActualTypeArguments();
+for (Type typeArgument : typeArguments) {
+if (typeArgument instanceof ParameterizedType) {
+return (Class) ((ParameterizedType) typeArgument).getRawType();
+} else {
+return (Class) typeArgument;
 }
-
+}
+}
 return Object.class;
 }
```

**After OD (correct):**
```diff
@@ -1,10 +1,17 @@
 public Class getGenericType(Field field) {
 Type generic = field.getGenericType();
-if (generic != null && generic instanceof ParameterizedType) {
-Type actual = ((ParameterizedType) generic).getActualTypeArguments()[0];
-return (Class) actual;
-//in case of nested generics we don't go deep
-}
-
+if (generic == null || !(generic instanceof ParameterizedType)) {
 return Object.class;
 }
+Type[] typeArguments = ((ParameterizedType) generic).getActualTypeArguments();
+for (Type typeArgument : typeArguments) {
+if (typeArgument instanceof ParameterizedType) {
+return (Class) ((ParameterizedType) typeArgument).getRawType();
+} else if (typeArgument instanceof Class) {
+return (Class) typeArgument;
+} else {
+return Object.class;
+}
+}
+return Object.class;
+}
```

**Ground truth fix:**
```diff
@@ -3,6 +3,10 @@
 if (generic != null && generic instanceof ParameterizedType) {
 Type actual = ((ParameterizedType) generic).getActualTypeArguments()[0];
+if (actual instanceof Class) {
 return (Class) actual;
+} else if (actual instanceof ParameterizedType) {
 //in case of nested generics we don't go deep
+return (Class) ((ParameterizedType) actual).getRawType();
+}
 }
 
```

---

