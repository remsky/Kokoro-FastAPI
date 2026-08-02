(function() {
function SiriWave(opt) {
  opt = opt || {};

  this.phase = 0;
  this.run = false;
  this._frameId = null;
  this._lastTs = null;
  this._boundDraw = this._draw.bind(this);

  // UI vars
  this.ratio = opt.ratio || window.devicePixelRatio || 1;
  this.height = this.ratio * (opt.height || 50);
  this.height_2 = this.height / 2;
  this.MAX = (this.height_2) - 4;

  // Constructor opt
  this.amplitude = opt.amplitude || 1;
  this.speed = opt.speed || 0.2;
  this.frequency = opt.frequency || 6;
  this.color = (function hex2rgb(hex){
    var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, function(m,r,g,b) { return r + r + g + g + b + b; });
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ?
    parseInt(result[1],16).toString()+','+parseInt(result[2], 16).toString()+','+parseInt(result[3], 16).toString()
    : null;
  })(opt.color || '#6366f1') || '99,102,241';

  // Canvas
  this.canvas = document.createElement('canvas');
  this.canvas.height = this.height;

  this.canvas.style.width = '100%';
  this.canvas.style.height = '100%';
  this.canvas.style.borderRadius = '4px';

  this.container = opt.container || document.body;
  this.container.appendChild(this.canvas);
  this.ctx = this.canvas.getContext('2d');

  // width must stay equal to canvas.width, _clear() erases exactly that box
  this.setWidth(opt.width || 320);

  // Start
  if (opt.autostart) {
    this.start();
  }
}

SiriWave.prototype._GATF_cache = {};
SiriWave.prototype._globAttFunc = function(x) {
  if (SiriWave.prototype._GATF_cache[x] == null) {
    SiriWave.prototype._GATF_cache[x] = Math.pow(4/(4+Math.pow(x,4)), 4);
  }
  return SiriWave.prototype._GATF_cache[x];
};

SiriWave.prototype._xpos = function(i) {
  return this.width_2 + i * this.width_4;
};

SiriWave.prototype._ypos = function(i, attenuation) {
  var att = (this.MAX * this.amplitude) / attenuation;
  return this.height_2 + this._globAttFunc(i) * att * Math.sin(this.frequency * i - this.phase);
};

SiriWave.prototype._drawLine = function(attenuation, color, width){
  this.ctx.moveTo(0,0);
  this.ctx.beginPath();
  this.ctx.strokeStyle = color;
  this.ctx.lineWidth = width || 1;

  var i = -2;
  while ((i += 0.01) <= 2) {
    var y = this._ypos(i, attenuation);
    if (Math.abs(i) >= 1.90) y = this.height_2;
    this.ctx.lineTo(this._xpos(i), y);
  }

  this.ctx.stroke();
};

SiriWave.prototype._clear = function() {
  this.ctx.globalCompositeOperation = 'destination-out';
  this.ctx.fillRect(0, 0, this.width, this.height);
  this.ctx.globalCompositeOperation = 'source-over';
};

SiriWave.prototype._draw = function(ts) {
  if (this.run === false) return;

  // speed is cycles/sec, a fixed per-frame step ran 2x on 120Hz displays
  var dt = 1/60;
  if (typeof ts === 'number') {
    if (this._lastTs !== null) dt = Math.min((ts - this._lastTs) / 1000, 0.05);
    this._lastTs = ts;
  }
  this.phase = (this.phase + 2*Math.PI*this.speed*dt) % (2*Math.PI);

  this._clear();
  this._drawLine(-2, 'rgba(' + this.color + ',0.1)');
  this._drawLine(-6, 'rgba(' + this.color + ',0.2)');
  this._drawLine(4, 'rgba(' + this.color + ',0.4)');
  this._drawLine(2, 'rgba(' + this.color + ',0.6)');
  this._drawLine(1, 'rgba(' + this.color + ',1)', 1.5);

  this._schedule();
};

SiriWave.prototype._schedule = function() {
  this._frameId = requestAnimationFrame(this._boundDraw);
};

SiriWave.prototype._cancelFrame = function() {
  if (this._frameId == null) return;
  cancelAnimationFrame(this._frameId);
  this._frameId = null;
};

SiriWave.prototype.start = function() {
  // not idempotent without this guard, a second call would spawn a second loop
  if (this.run) return;
  this.phase = 0;
  this._lastTs = null;
  this.run = true;
  this._draw();
};

SiriWave.prototype.stop = function() {
  this.phase = 0;
  this._lastTs = null;
  this.run = false;
  this._cancelFrame();
};

SiriWave.prototype.setWidth = function(cssWidth) {
  var px = Math.max(1, Math.round(this.ratio * (cssWidth || 0)));
  if (px === this.width) return;
  this.width = px;
  this.width_2 = px / 2;
  this.width_4 = px / 4;
  this.canvas.width = px;
};

SiriWave.prototype.dispose = function() {
  this.stop();
  if (this.canvas && this.canvas.parentNode) {
    this.canvas.parentNode.removeChild(this.canvas);
  }
};

SiriWave.prototype.setSpeed = function(v) {
  this.speed = v;
};

SiriWave.prototype.setNoise = SiriWave.prototype.setAmplitude = function(v) {
  this.amplitude = Math.max(Math.min(v, 1), 0);
};

if (typeof define === 'function' && define.amd) {
  define(function(){ return SiriWave; });
  return;
};
window.SiriWave = SiriWave;
})();