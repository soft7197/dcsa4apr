# Overfitting Detector Gains - CodeLlama Exp 1

Overfitting detector gain bugs: 2

These bugs had **no correct patches before OD refinement**. All pre-refinement patches were overfitting ??they passed tests but did not actually fix the root cause. OD refinement produced correct patches for these bugs.

---

## Jsoup-13

**Root cause:** The bug is that Node.hasAttr() only checks for a literal attribute key and does not specially handle the synthetic "abs:" prefix, which should report true when the underlying attribute exists and resolves to a non-empty absolute URL.

**Why pre-patch overfits:** The pre-patch fixes the main missing "abs:" handling and matches the intended behavior on normal inputs, but it is not fully robust because it directly calls absUrl() without guarding against exceptions. On reasonable inputs outside the provided tests, such as malformed base URIs or URL values that make absUrl() throw, hasAttr("abs:...") would incorrectly throw instead of returning false.

**OD refinement (NECESSARY):** The refinement added exception handling around absUrl(attributeKey.substring(4)) for "abs:" attributes, returning false instead of propagating an exception. This is a real semantic change: the pre-patch can throw on malformed URLs or URI edge cases, whereas hasAttr() should behave as a boolean predicate and not fail for such inputs.

**Verdict:** Because the refinement change is NECESSARY and adds missing behavior not otherwise present in the pre-patch, the original patch was not semantically complete. It passed the tests but remained overfitted due to unhandled exceptional cases in the new "abs:" path.

**Before OD (overfitting):**
```diff
@@ -2,4 +2,8 @@
 Validate.notNull(attributeKey);
 
+if (attributeKey.startsWith("abs:")) {
+return absUrl(attributeKey.substring(4)).length() > 0;
+}
+
 return attributes.hasKey(attributeKey);
 }
```

**After OD (correct):**
```diff
@@ -2,4 +2,12 @@
 Validate.notNull(attributeKey);
 
+if (attributeKey.startsWith("abs:")) {
+try {
+return absUrl(attributeKey.substring(4)).length() > 0;
+} catch (Exception e) {
+return false;
+}
+}
+
 return attributes.hasKey(attributeKey);
 }
```

**Ground truth fix:**
```diff
@@ -2,4 +2,9 @@
 Validate.notNull(attributeKey);
 
+if (attributeKey.toLowerCase().startsWith("abs:")) {
+String key = attributeKey.substring("abs:".length());
+if (attributes.hasKey(key) && !absUrl(key).equals(""))
+return true;
+}
 return attributes.hasKey(attributeKey);
 }
```

---

## Jsoup-57

**Root cause:** The bug is caused by removing from the backing map directly while iterating over its keySet, which triggers a ConcurrentModificationException; removal must be done through the iterator.

**Why pre-patch overfits:** The pre-patch only partially fixes the root cause: it avoids concurrent modification by using `it.remove()`, but it stops after the first match. On reasonable inputs containing multiple attributes whose names differ only by case, it would leave some matching attributes behind, whereas the correct behavior is to remove them all.

**OD refinement (NECESSARY):** The OD refinement removed the early `return` after `it.remove()`, changing the method from removing only the first case-insensitive match to removing all matching keys during iteration. This is semantically important because the method contract and the ground-truth fix imply that every attribute whose key equals the target ignoring case should be removed, not just the first one encountered.

**Verdict:** Because the refinement change is NECESSARY and adds missing logic to continue scanning and remove all case-insensitive matches, the pre-patch is an overfitting partial fix rather than a fully correct one.

**Before OD (overfitting):**
```diff
@@ -5,6 +5,8 @@
 for (Iterator<String> it = attributes.keySet().iterator(); it.hasNext(); ) {
 String attrKey = it.next();
-if (attrKey.equalsIgnoreCase(key))
-attributes.remove(attrKey);
+if (attrKey.equalsIgnoreCase(key)) {
+it.remove();
+return;
 }
 }
+}
```

**After OD (correct):**
```diff
@@ -2,9 +2,13 @@
 Validate.notEmpty(key);
 if (attributes == null)
+{
 return;
+}
+String attrKey = null;
 for (Iterator<String> it = attributes.keySet().iterator(); it.hasNext(); ) {
-String attrKey = it.next();
-if (attrKey.equalsIgnoreCase(key))
-attributes.remove(attrKey);
+attrKey = it.next();
+if (attrKey.equalsIgnoreCase(key)) {
+it.remove();
 }
 }
+}
```

**Ground truth fix:**
```diff
@@ -6,5 +6,5 @@
 String attrKey = it.next();
 if (attrKey.equalsIgnoreCase(key))
-attributes.remove(attrKey);
+it.remove();
 }
 }
```

---

