"""Render database DDL schema and ER relation (stored as JSON) into PNG images.
Usage:
	python preprocessing/render_schema_to_image.py [options]

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import json
import re
import logging
from utils.my_logger import setup_logging
import argparse
from graphviz import Digraph


setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def render_single_image(text, db_id, output_dir, font_size=22,
						padding=4, max_width=1024,
						bg_color=(255, 255, 255), text_color=(0, 0, 0),
						line_spacing=2,
						preserve_paragraphs=True):
	"""Render ``text`` onto a single image saved as ``<output_dir>/<db_id>.png``.

	The image is sized to fit the content tightly: width is capped at
	``max_width`` while the height grows with the number of lines. Text is
	wrapped so that every line fits within the usable width; words longer than
	a line are split character by character so nothing is ever clipped.
	"""

	font_path = "./font/arial.ttf"

	# Try to load the TrueType font; fall back to PIL's bitmap default on any
	# error (e.g. missing font file) rather than crashing.
	try:
		font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
	except Exception as e:
		logger.info(f"load font failed: {e}")

	# 1. Retrieve the font's metrics to compute an accurate per-line height.
	ascent, descent = font.getmetrics()
	line_height = ascent + descent
	usable_width = max_width - padding * 2

	# 2. Split the text into paragraphs. When ``preserve_paragraphs`` is on the
	# original blank lines are kept; otherwise the whole text is one paragraph.
	if preserve_paragraphs:
		paragraphs = text.split('\n')
	else:
		paragraphs = [text]

	all_lines = []

	for para in paragraphs:
		# Keep blank paragraphs as blank lines in the output.
		if not para.strip():
			all_lines.append('')
			continue

		# Tokenise the paragraph into words (whitespace as separator).
		words = para.split()
		if not words:
			all_lines.append('')
			continue

		# 3. Build each line word by word, measuring with exact pixel widths.
		current_line = []
		current_width = 0

		for word in words:
			# Measure the width of the word itself.
			word_bbox = font.getbbox(word)
			word_width = word_bbox[2] - word_bbox[0]

			# Width of a single space, used as the gap between words.
			space_bbox = font.getbbox(' ')
			space_width = space_bbox[2] - space_bbox[0]

			# Width if this word were appended to the current line.
			if current_line:
				test_width = current_width + space_width + word_width
			else:
				test_width = word_width

			if test_width <= usable_width:
				# The word fits: add it to the current line.
				current_line.append(word)
				current_width = test_width
			else:
				# The current line is full, so flush it first.
				if current_line:
					all_lines.append(' '.join(current_line))
				else:
					# A single word already longer than the line: emit it alone
					# (rare) and remember its width for accurate wrapping.
					all_lines.append(word)
					current_width = word_width
				current_line = []
				current_width = 0

				# Now try to start a new line with this word.
				if word_width <= usable_width:
					current_line.append(word)
					current_width = word_width
				else:
					# The word itself is wider than the line: break it into
					# individual characters that each fit within the line.
					chars = list(word)
					sub_line = []
					sub_width = 0
					for char in chars:
						char_bbox = font.getbbox(char)
						char_width = char_bbox[2] - char_bbox[0]
						if sub_width + char_width <= usable_width:
							sub_line.append(char)
							sub_width += char_width
						else:
							# Newline needed: flush the current character run.
							if sub_line:
								all_lines.append(''.join(sub_line))
							sub_line = [char]
							sub_width = char_width
					if sub_line:
						all_lines.append(''.join(sub_line))

		# Snapshot the finalized current line (the last one of the paragraph).
		if current_line:
			all_lines.append(' '.join(current_line))

	# 4. Measure the exact pixel size of every wrapped line.
	line_heights = []
	line_widths = []
	max_width_actual = 0

	for line in all_lines:
		if line:
			bbox = font.getbbox(line)
			width = bbox[2] - bbox[0]
			height = bbox[3] - bbox[1]
		else:
			width = 0
			height = line_height  # Blank lines use the standard line height.

		line_widths.append(width)
		line_heights.append(height)
		max_width_actual = max(max_width_actual, width)

	# 5. Compute the image size so the content fits tightly.
	total_text_height = sum(line_heights) + (len(all_lines) - 1) * line_spacing
	image_width = int(min(max_width_actual + padding * 2, max_width))
	image_height = int(total_text_height + padding * 2)

	# 6. Create the canvas and draw each line from top to bottom.
	img = Image.new('RGB', (image_width, image_height), bg_color)
	draw = ImageDraw.Draw(img)

	y = padding
	for i, line in enumerate(all_lines):
		if line:
			# Draw the line at its baseline position (top-left anchored).
			bbox = font.getbbox(line)
			draw.text((padding, y), line, font=font, fill=text_color)
		y += line_heights[i] + line_spacing

	save_path = f"{output_dir}/{db_id}.png"
	img.save(save_path)

	logger.info(f"render Database[{db_id}] DDL schema to a {image_width} x {image_height} image")


def render_multi_image(text, db_id, output_dir, font_size=22,
					   padding=4, max_width=1024,
					   bg_color=(255, 255, 255), text_color=(0, 0, 0),
					   line_spacing=2,
					   preserve_paragraphs=True,
					   max_image_height=1024):
	"""Render ``text`` onto a series of images, each capped at ``max_image_height``.

	Content taller than a single page is split across multiple images saved as
	``<output_dir>/<db_id>_part<N>.png``. This keeps long schemas readable
	instead of producing one impossibly tall image.
	"""

	font_path = "./font/arial.ttf"

	# Try the TrueType font, otherwise fall back to the default (bitmap) font.
	try:
		font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
	except Exception as e:
		print(f"load font failed: {e}")
		font = ImageFont.load_default()

	# 2. Retriev the font metrics to compute an accurate line height.
	ascent, descent = font.getmetrics()
	line_height = ascent + descent
	usable_width = max_width - padding * 2

	# 3. Split into paragraphs and wrap every line (same logic as the single
	# image renderer, kept inlined for readability).
	if preserve_paragraphs:
		paragraphs = text.split('\n')
	else:
		paragraphs = [text]

	all_lines = []

	for para in paragraphs:
		# Preserve blank paragraphs as blank lines.
		if not para.strip():
			all_lines.append('')
			continue

		words = para.split()
		if not words:
			all_lines.append('')
			continue

		current_line = []
		current_width = 0

		for word in words:
			word_bbox = font.getbbox(word)
			word_width = word_bbox[2] - word_bbox[0]

			space_bbox = font.getbbox(' ')
			space_width = space_bbox[2] - space_bbox[0]

			# Test whether the word fits on the current line.
			if current_line:
				test_width = current_width + space_width + word_width
			else:
				test_width = word_width

			if test_width <= usable_width:
				current_line.append(word)
				current_width = test_width
			else:
				# Line is full: flush it, then retry the word on a fresh line.
				if current_line:
					all_lines.append(' '.join(current_line))
				else:
					all_lines.append(word)
					current_width = word_width
				current_line = []
				current_width = 0

				if word_width <= usable_width:
					current_line.append(word)
					current_width = word_width
				else:
					# Overlong word: split it character by character.
					chars = list(word)
					sub_line = []
					sub_width = 0
					for char in chars:
						char_bbox = font.getbbox(char)
						char_width = char_bbox[2] - char_bbox[0]
						if sub_width + char_width <= usable_width:
							sub_line.append(char)
							sub_width += char_width
						else:
							if sub_line:
								all_lines.append(''.join(sub_line))
							sub_line = [char]
							sub_width = char_width
					if sub_line:
						all_lines.append(''.join(sub_line))

		# Close out the final line of the paragraph.
		if current_line:
			all_lines.append(' '.join(current_line))

	# 4. Measure the exact pixel dimensions of every wrapped line.
	line_heights = []
	line_widths = []
	max_width_actual = 0

	for line in all_lines:
		if line:
			bbox = font.getbbox(line)
			width = bbox[2] - bbox[0]
			height = bbox[3] - bbox[1]
		else:
			width = 0
			height = line_height

		line_widths.append(width)
		line_heights.append(height)
		max_width_actual = max(max_width_actual, width)

	# 5. Work out how many consecutive lines fit inside a single page
	# (the usable content height is the page height minus the padding).
	max_content_height = max_image_height - padding * 2
	lines_per_page = 0
	current_height = 0

	for i, height in enumerate(line_heights):
		if i > 0:
			current_height += line_spacing
		if current_height + height <= max_content_height:
			current_height += height
			lines_per_page += 1
		else:
			break

	# Make sure at least one line is rendered, even if it slightly exceeds the
	# page height (avoids an infinite-empty output).
	if lines_per_page == 0:
		lines_per_page = 1
		print(f"⚠️ warning: ({line_heights[0]}px) over setting height")

	# 6. Split the content into pages and render each one.
	total_lines = len(all_lines)
	total_pages = (total_lines + lines_per_page - 1) // lines_per_page

	images = []
	image_paths = []

	for page_idx in range(total_pages):
		start_idx = page_idx * lines_per_page
		end_idx = min(start_idx + lines_per_page, total_lines)
		page_lines = all_lines[start_idx:end_idx]
		page_heights = line_heights[start_idx:end_idx]
		page_widths = line_widths[start_idx:end_idx]

		# Height of the current page's content block.
		page_content_height = sum(page_heights) + (len(page_lines) - 1) * line_spacing
		page_image_height = int(page_content_height + padding * 2)
		# Ensure the height never exceeds the requested maximum (a single
		# overlong line may still nudge it over by design).
		page_image_height = min(page_image_height, max_image_height)

		# Width is fixed for every page.
		page_image_width = int(min(max_width_actual + padding * 2, max_width))

		# Create the page canvas.
		img = Image.new('RGB', (page_image_width, page_image_height), bg_color)
		draw = ImageDraw.Draw(img)

		# Draw the page's lines from top to bottom.
		y = padding
		for i, line in enumerate(page_lines):
			if line:
				draw.text((padding, y), line, font=font, fill=text_color)
			y += page_heights[i] + line_spacing

		# Record the page for later saving.
		page_save_path = f"{output_dir}/{db_id}_part{str(page_idx+1)}.png"

		image_paths.append(page_save_path)
		images.append(img)

	logger.info(f"render Database[{db_id}] DDL schema to {len(images)} images")
	

	# Save every rendered page to disk.
	for i, img in enumerate(images):
		img.save(image_paths[i])


def render_ddl_schema_to_image(args):
	"""Load the schema JSON and render every database described by it."""

	# Load the schema file (a dict of db_id -> {table -> {...}}).
	with open(args.input_file, 'r', encoding='utf-8') as f:
		data = json.load(f)

	for db_id in data:
		# Assemble the textual description of the current database.
		content = ''
		content += '#database : ' + db_id + '\n\n'
		fk_list = []
		for table in data[db_id]:
			schema = data[db_id][table]['schema']
			# Flatten newlines and collapse runs of whitespace, then add commas
			# around them for cleaner word wrapping.
			schema = re.sub(r'\n', ' ', schema)
			schema = re.sub(r'\s+', ' ', schema)
			schema = schema.replace(',', ' ,')
			# Drop nullability keywords to keep the schema compact.
			schema = schema.replace('not null', '').replace('null', '')
			content += schema + '\n\n'

			# Collect each foreign key exactly once (skip duplicates/None).
			for item in data[db_id][table]['foreign_keys']:
				fk = item['constraint']
				if fk not in fk_list and 'None' not in fk:
					fk_list.append(fk)

		# Optionally append the discovered foreign-key relationships.
		if args.render_relation:
			content += '#relationship :' + '\n'
			for fk in fk_list:
				fk = fk.replace('=', ' = ')
				content += fk + '\n'

		# Render according to the selected mode: multi-page or single image.
		if args.render_mode == 'multi':
			render_multi_image(
				content,
				db_id,
				args.output_dir,
				font_size=22,
				max_width=1024,
				padding=6,
				line_spacing=3,
				max_image_height=2048
			)
		else:
			render_single_image(
				content,
				db_id,
				args.output_dir,
				font_size=20,
				max_width=1024,
				padding=5,
				line_spacing=4
			)


def render_er_relation_to_image(args):
	"render database ER relationship information to one image "

	with open(args.input_file, 'r', encoding='utf-8') as f:
		data = json.load(f)
	
	edge_colors = [
		'#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', 
		'#E67E22', '#34495E', '#E74C3C', '#2ECC71', '#F1C40F', '#1ABC9C',
		'#E67E22', '#3498DB', '#9B59B6', '#2ECC71', '#F39C12', '#E74C3C',
		'#3498DB', '#1ABC9C', '#E67E22', '#9B59B6', '#F1C40F', '#2ECC71',
		'#E74C3C', '#3498DB', '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22',
		'#34495E', '#2ECC71', '#E74C3C', '#F1C40F', '#3498DB', '#9B59B6',
		'#E67E22', '#1ABC9C', '#F39C12', '#2ECC71', '#E74C3C', '#3498DB'
	]
	
	for db_id in data:
		dot = Digraph(comment='E-R Diagram', format='png')
		dot.attr(rankdir='TB', bgcolor='white', rank='min', center='1')
		dot.attr('node', shape='none', fontname='Helvetica')

		color_index = 0
		tables = data[db_id]
		# # One node per table, labelled with its key columns only.
		for table in tables:
			# get primary_keys and 
			pk_columns = tables[table]['primary_keys']
			
			# get forgein keys
			fk_columns = []
			for item in tables[table]['foreign_keys']:
				fk_from = item['from']
				if '.None' in item['constraint']:
					continue
				fk = fk_from.split('.')[1].strip()
				fk_columns.append(fk)

			# add table tag
			label = f'''<
			<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
			<TR><TD BGCOLOR="#4472C4" ALIGN="CENTER"><FONT COLOR="white"><B>{table}</B></FONT></TD></TR>
			'''
	
			important_cols = list(set(pk_columns + fk_columns))
			if important_cols:
				for col_name in important_cols:
					is_pk = col_name in pk_columns
					is_fk = col_name in fk_columns
					marks = []
					if is_pk:
						marks.append('🔑')
					if is_fk:
						marks.append('🔗')
					
					if marks:
						mark_str = f' <FONT POINT-SIZE="10">{''.join(marks)}</FONT>'
					else:
						mark_str = ''
			
					label += f'<TR><TD ALIGN="CENTER">{col_name}{mark_str}</TD></TR>\n'
			else:
				label += f'<TR><TD ALIGN="CENTER"><I>""</I></TD></TR>\n'
			label += '</TABLE>>'
			dot.node(table, label=label)
	
		
		relationships = []
		for table in tables:
			for item in tables[table]['foreign_keys']:
				fk_from = item['from']
				fk_to = item['to']
				if '.None' in item['constraint']:
					continue
				from_col = fk_from.split('.')[1].strip()
				to_col =  fk_to.split('.')[1].strip()
				ref_table = fk_to.split('.')[0].strip()

				color = edge_colors[color_index % len(edge_colors)]
				color_index += 1
				dot.edge(table, ref_table, 
							color=color,
							penwidth='2.0',
							arrowsize='0.8')

				if from_col and to_col:
					relationships.append({
					'from_table': table,
					'from_col': from_col,
					'to_table': ref_table,
					'to_col': to_col,
					'color': color
					})

		rel_label = '<'
		rel_label += '<TABLE BORDER="0" CELLSPACING="2" CELLPADDING="2">'
		rel_label += f'<TR><TD COLSPAN="3" BGCOLOR="#F0F0F0" ALIGN="CENTER"><B>relationships of {db_id}</B></TD></TR>'
		
		for rel in relationships:
			rel_label += f'<TR>'
			rel_label += f'<TD ALIGN="RIGHT"><FONT COLOR="{rel["color"]}"><B>{rel["from_table"]}.{rel["from_col"]}</B></FONT></TD>'
			rel_label += f'<TD ALIGN="CENTER"><FONT COLOR="{rel["color"]}"> = </FONT></TD>'
			rel_label += f'<TD ALIGN="LEFT"><FONT COLOR="{rel["color"]}"><B>{rel["to_table"]}.{rel["to_col"]}</B></FONT></TD>'
			rel_label += f'</TR>'
		
		rel_label += '</TABLE>>'
		
		dot.node('relationships', label=rel_label, shape='plaintext', fontname='Helvetica')
		with dot.subgraph() as s:
			s.attr(rank='source', rankdir='TB')
			s.node('relationships')
		
		# save
		output = f'{args.output_dir}/{db_id}'
		dot.render(output, view=False, cleanup=True, format='png')
		logger.info(f"render Database[{db_id}] ER relation to  a image")
		del dot


def parser():
	"""Define and return the command-line argument parser."""
	parser = argparse.ArgumentParser("")
	parser.add_argument('--input_file', type=str, default="./data/BIRD/dev/dev_ddl_schema.json")
	parser.add_argument('--render_mode', type=str, default='multi')
	parser.add_argument('--render_relation', type=bool, default=False)
	parser.add_argument('--output_dir', type=str, default='GV_images/dev')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parser()
	# render_ddl_schema_to_image(args)
	render_er_relation_to_image(args)
