create extension if not exists pg_trgm with schema extensions;

create index if not exists coarse_unit_label_lower_idx
  on semantic.coarse_unit using btree (lower(label));

create index if not exists coarse_unit_label_lower_trgm_idx
  on semantic.coarse_unit using gin (lower(label) gin_trgm_ops);

drop function if exists public.coarse_unit_label_exact(text, int);

create function public.coarse_unit_label_exact(
  q text,
  n int default 20
)
returns table (
  id bigint,
  kind text,
  label text,
  pos text,
  chinese_def text,
  chinese_criteria text,
  chinese_label text,
  pattern jsonb,
  status text
)
language plpgsql
stable
as $$
declare
  q_norm text;
  limit_n int;
begin
  q_norm := nullif(btrim(q), '');
  if q_norm is null then
    return;
  end if;

  limit_n := greatest(1, least(coalesce(n, 20), 20));

  return query
  select c.id,
         c.kind,
         c.label,
         c.pos,
         c.chinese_def,
         c.chinese_criteria,
         c.chinese_label,
         c.pattern,
         c.status
  from semantic.coarse_unit as c
  where lower(c.label) = lower(q_norm)
  order by c.label asc, c.id asc
  limit limit_n;
end;
$$;

drop function if exists public.coarse_unit_label_contains(text, int);

create function public.coarse_unit_label_contains(
  q text,
  n int default 20
)
returns table (
  id bigint,
  kind text,
  label text,
  pos text,
  chinese_def text,
  chinese_criteria text,
  chinese_label text,
  pattern jsonb,
  status text
)
language plpgsql
stable
as $$
declare
  q_norm text;
  limit_n int;
begin
  q_norm := nullif(btrim(q), '');
  if q_norm is null then
    return;
  end if;

  q_norm := lower(q_norm);
  limit_n := greatest(1, least(coalesce(n, 20), 20));

  return query
  select c.id,
         c.kind,
         c.label,
         c.pos,
         c.chinese_def,
         c.chinese_criteria,
         c.chinese_label,
         c.pattern,
         c.status
  from semantic.coarse_unit as c
  where lower(c.label) like '%' || q_norm || '%'
  order by strpos(lower(c.label), q_norm) asc,
           char_length(c.label) asc,
           c.label asc,
           c.id asc
  limit limit_n;
end;
$$;
