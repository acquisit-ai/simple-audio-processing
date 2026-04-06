# coarse_unit label search

## Added

- `public.coarse_unit_label_exact(q text, n int default 20)`
- `public.coarse_unit_label_contains(q text, n int default 20)`
- `pg_trgm` extension in `extensions` schema if it is not already installed
- `semantic.coarse_unit(lower(label))` trigram GIN index for contains search
- reuse of existing `semantic.coarse_unit(lower(label))` btree index for exact search

## Not changed

- no business table columns in `semantic.coarse_unit`
- no RLS or policy changes
- no exposed schema changes
- no Edge Function, api schema, or extra backend service

## Rollback

Run these statements in a new rollback migration if needed:

```sql
drop function if exists public.coarse_unit_label_contains(text, int);
drop function if exists public.coarse_unit_label_exact(text, int);
drop index if exists semantic.coarse_unit_label_lower_trgm_idx;
```

Keep `semantic.coarse_unit_label_lower_idx` in place unless you have confirmed it was created only for this change, because it already existed before this migration in the current database.

实现了两个可通过 Supabase RPC 调用的 Postgres 查询函数，用于查询 semantic.coarse_unit.label：

1. public.coarse_unit_label_exact(q text, n int default 20)
   作用：对 semantic.coarse_unit.label 做大小写不敏感的精确匹配，等价于 lower(label) = lower(trim(q))。
   返回：最多 20 条，返回 id、kind、label、pos、chinese_def、chinese_criteria、chinese_label、pattern、status。
   规则：q 为 NULL 或空白时返回空结果；n 会被限制在 1..20；排序为 label asc, id asc。

2. public.coarse_unit_label_contains(q text, n int default 20)
   作用：对 semantic.coarse_unit.label 做大小写不敏感的包含匹配，等价于 lower(label) like '%' || lower(trim(q)) || '%'。
   返回：最多 20 条，返回 id、kind、label、pos、chinese_def、chinese_criteria、chinese_label、pattern、status。
   规则：q 为 NULL 或空白时返回空结果；n 会被限制在 1..20；排序优先匹配位置更靠前、label 更短，再按 label asc, id asc。

调用方式：
supabase.rpc('coarse_unit_label_exact', { q: 'ACE', n: 20 })
supabase.rpc('coarse_unit_label_contains', { q: 'ace', n: 20 })

补充：

- 只查询 semantic.coarse_unit.label
- 不改业务表结构
- 不改 RLS / policy
- 精确查询走 lower(label) btree 索引
- 包含查询走 lower(label) trigram GIN 索引
