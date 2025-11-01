create schema if not exists semantic;
-- ===========================================================
--  粗粒度词典表结构
--  说明：
--    1. 保留 fine_unit 中除 meta / external_key 以外的核心字段。
--    2. 新增 fine_unit_ids（bigint[]）关联多个细粒度释义。
--    3. original_defs 使用 text[] 保存所有原始细粒度释义文本。
--    4. english_def / chinese_def 存储粗粒度聚合后的双语解释。
--    5. 通过触发器确保 fine_unit_ids 内的 id 均存在于 fine_unit。
-- ===========================================================

create table if not exists semantic.coarse_unit (
  id             bigserial primary key,                        -- 粗粒度释义唯一 ID
  kind           text not null,                                -- 与 fine_unit.kind 对齐
  label          text not null,                                -- 粗粒度标签
  lang           text not null default 'en',                   -- 语言
  pos            text,                                         -- 粗粒度词性
  def            text,                                         -- 可选通用释义（保持与原表结构一致）
  english_def    text,                                         -- 粗粒度英文解释
  chinese_def    text,                                         -- 粗粒度中文解释
  chinese_criteria text,                                       -- 中文判据说明（包含/排除的依据）
  chinese_label  text,                                         -- 中文标签/直译
  pattern        jsonb,                                        -- 可选模式信息
  status         text not null default 'active',               -- 状态
  created_at     timestamptz not null default now(),           -- 创建时间
  updated_at     timestamptz not null default now(),           -- 更新时间
  fine_unit_ids  bigint[] not null,                            -- 关联的细粒度 fine_unit.id 列表
  original_defs  text[] not null,                              -- 所有关联 fine_unit 的原始释义
  constraint coarse_unit_kind_check check (
    kind = any (array['word_sense','phrase_sense','grammar_rule'])
  )
) tablespace pg_default;

create index if not exists ix_coarse_kind
  on semantic.coarse_unit using btree (kind) tablespace pg_default;

create index if not exists ix_coarse_label
  on semantic.coarse_unit using btree (label) tablespace pg_default;

-- ===========================================================
--  校验触发器：确保 fine_unit_ids 中的每个 id 都存在
-- ===========================================================

create or replace function semantic.coarse_unit_validate_fine_ids()
returns trigger
language plpgsql
as $$
declare
  missing_id bigint;
begin
  -- 若存在未在 fine_unit 中出现的 id，则阻止写入
  select x
    into missing_id
    from unnest(new.fine_unit_ids) as x
   where not exists (
         select 1 from semantic.fine_unit f where f.id = x
   )
   limit 1;

  if missing_id is not null then
    raise exception 'fine_unit id % does not exist', missing_id;
  end if;

  -- 每次写入或更新都刷新更新时间
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists trg_coarse_unit_validate on semantic.coarse_unit;

create trigger trg_coarse_unit_validate
before insert or update on semantic.coarse_unit
for each row execute function semantic.coarse_unit_validate_fine_ids();
