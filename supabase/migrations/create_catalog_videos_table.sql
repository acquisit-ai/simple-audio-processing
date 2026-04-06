
-- ============================================
-- 2) 主表：videos（含“留空自动生成/显式传入”两用主键）
-- ============================================
create table if not exists catalog.videos (
  video_id             uuid primary key default gen_random_uuid(),         -- 支持留空自动生成或显式传入
  upload_user_id       uuid not null,                                      -- 上传者（auth.users.id）
  created_at           timestamptz not null default now(),                 -- 默认 UTC
  updated_at           timestamptz not null default now(),                 -- 由触发器更新

  title                text not null,                                      -- 标题
  description          text,                                               -- 描述
  raw_file_reference   text not null,                                      -- 原始对象位置/键（如 GCS 路径 + 扩展名）
  status               catalog.video_status not null default 'pending_upload', -- 总体状态
  version              bigint not null default 1,                          -- 并发控制版本号（乐观锁）
  media_status         catalog.stage_status  not null default 'pending',   -- 媒体阶段
  analysis_status      catalog.stage_status  not null default 'pending',   -- AI 阶段
  media_job_id         text,                                               -- 最近一次媒体流水线任务ID
  media_emitted_at     timestamptz,                                        -- 最近一次媒体结果回写时间
  analysis_job_id      text,                                               -- 最近一次 AI 任务ID
  analysis_emitted_at  timestamptz,                                        -- 最近一次 AI 结果回写时间

  -- 上传完成后补写的原始媒体属性
  raw_file_size        bigint check (raw_file_size > 0),                   -- 字节
  raw_resolution       text,                                               -- 如 3840x2160
  raw_bitrate          integer,                                            -- kbps

  -- 媒体转码完成后补写
  duration_micros      bigint,                                             -- 微秒
  encoded_resolution   text,
  encoded_bitrate      integer,
  thumbnail_url        text,
  hls_master_playlist  text,

  -- AI 分析完成后补写
  difficulty           text,
  summary              text,
  tags                 text[],                                             -- 标签数组（配 GIN 索引）

  -- 可见性层字段（Safety 写入）
  visibility_status   text not null default 'public',                     -- 可见性状态 public/unlisted/private
  publish_at          timestamptz,                                        -- 发布时间（UTC），可为空

  raw_subtitle_url     text,                                               -- 原始字幕/ASR 输出
  error_message        text                                                -- 最近失败/拒绝原因
);

comment on table catalog.videos is '视频主表：记录上传者、状态流转、媒体与AI分析产物等';

-- 字段注释（逐列）
comment on column catalog.videos.video_id            is '主键：UUID（默认 gen_random_uuid()）。可显式传入自生成 UUID 覆盖默认';
comment on column catalog.videos.upload_user_id      is '上传者用户ID（auth.users.id），受 RLS 策略约束';
comment on column catalog.videos.created_at          is '记录创建时间（timestamptz, 默认 now()）';
comment on column catalog.videos.updated_at          is '最近更新时间（timestamptz），由触发器在 UPDATE 时写入 now()';

comment on column catalog.videos.title               is '视频标题（必填）';
comment on column catalog.videos.description         is '视频描述（可选，长文本）';
comment on column catalog.videos.raw_file_reference  is '原始对象位置（如 gs://bucket/path/file.mp4）';
comment on column catalog.videos.status              is '总体状态：pending_upload→processing→ready/published 或 failed/rejected/archived';
comment on column catalog.videos.version             is '乐观锁版本号：每次业务更新自增，用于并发控制与事件 version';
comment on column catalog.videos.media_status        is '媒体阶段状态：pending/processing/ready/failed（转码/封面等）';
comment on column catalog.videos.analysis_status     is 'AI 阶段状态：pending/processing/ready/failed（ASR/标签/摘要等）';
comment on column catalog.videos.media_job_id        is '最近一次媒体流水线任务ID（用于幂等与事件序）';
comment on column catalog.videos.media_emitted_at    is '最近一次媒体任务完成时间（用于拒绝旧事件）';
comment on column catalog.videos.analysis_job_id     is '最近一次 AI 任务ID（用于幂等与事件序）';
comment on column catalog.videos.analysis_emitted_at is '最近一次 AI 任务完成时间（用于拒绝旧事件）';

comment on column catalog.videos.raw_file_size       is '原始文件大小（字节，>0）';
comment on column catalog.videos.raw_resolution      is '原始分辨率（如 3840x2160）';
comment on column catalog.videos.raw_bitrate         is '原始码率（kbps）';

comment on column catalog.videos.duration_micros     is '视频时长（微秒）';
comment on column catalog.videos.encoded_resolution  is '主转码分辨率（如 1920x1080）';
comment on column catalog.videos.encoded_bitrate     is '主转码码率（kbps）';
comment on column catalog.videos.thumbnail_url       is '主缩略图 URL/路径';
comment on column catalog.videos.hls_master_playlist is 'HLS 主清单（master.m3u8）URL/路径';

comment on column catalog.videos.difficulty          is 'AI 评估难度（自由文本，可后续枚举化）';
comment on column catalog.videos.summary             is 'AI 生成摘要';
comment on column catalog.videos.tags                is 'AI 生成标签（text[]，使用 GIN 索引提升包含查询）';
comment on column catalog.videos.visibility_status   is '可见性状态：public/unlisted/private，由 Safety 服务写入';
comment on column catalog.videos.publish_at          is '发布时间（UTC），当视频上架时写入';

comment on column catalog.videos.raw_subtitle_url    is '原始字幕/ASR 输出 URL/路径';
comment on column catalog.videos.error_message       is '最近一次失败/拒绝原因（排障/审计）';

-- ============================================
-- 3) 外键（引用 Supabase Auth 用户，禁止级联删除）
-- ============================================
do $$
begin
  if not exists (
    select 1
      from pg_constraint
     where conname = 'videos_upload_user_fkey'
       and conrelid = 'catalog.videos'::regclass
  ) then
    alter table catalog.videos
      add constraint videos_upload_user_fkey
      foreign key (upload_user_id)
      references auth.users(id)
      on update cascade
      on delete restrict;
  end if;
end$$;

comment on constraint videos_upload_user_fkey on catalog.videos
  is '外键：绑定到 auth.users(id)；更新级联，删除限制（不随用户删除而删除视频）';

-- ============================================
-- 4) 索引（含显式 schema 前缀的注释，避免 42P01）
-- ============================================
create index if not exists videos_status_idx
  on catalog.videos (status);
comment on index catalog.videos_status_idx            is '按总体状态过滤（队列/面板）';

create index if not exists videos_media_status_idx
  on catalog.videos (media_status);
comment on index catalog.videos_media_status_idx      is '按媒体阶段状态过滤（监控转码队列）';

create index if not exists videos_analysis_status_idx
  on catalog.videos (analysis_status);
comment on index catalog.videos_analysis_status_idx   is '按分析阶段状态过滤（监控AI队列）';

create index if not exists videos_tags_gin_idx
  on catalog.videos using gin (tags);
comment on index catalog.videos_tags_gin_idx          is '标签数组的 GIN 索引，支持多标签检索';

create index if not exists videos_upload_user_idx
  on catalog.videos (upload_user_id);
comment on index catalog.videos_upload_user_idx       is '按上传者查找其视频列表';

create index if not exists videos_created_at_idx
  on catalog.videos (created_at);
comment on index catalog.videos_created_at_idx        is '按创建时间排序/分页（Feed/归档）';

-- ============================================
-- 5) 更新时间戳触发器（自动维护 updated_at = now()）
-- ============================================
create or replace function catalog.tg_set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;
comment on function catalog.tg_set_updated_at() is '触发器函数：在 UPDATE 时把 updated_at 写为 now()';

do $$
begin
  if not exists (
    select 1 from pg_trigger where tgname = 'set_updated_at_on_videos'
  ) then
    create trigger set_updated_at_on_videos
      before update on catalog.videos
      for each row execute function catalog.tg_set_updated_at();
  end if;
end$$;
comment on trigger set_updated_at_on_videos on catalog.videos
  is '更新 catalog.videos 任意列时自动刷新 updated_at';
