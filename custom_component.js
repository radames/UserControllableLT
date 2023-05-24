(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))r(o);new MutationObserver(o=>{for(const i of o)if(i.type==="childList")for(const a of i.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function n(o){const i={};return o.integrity&&(i.integrity=o.integrity),o.referrerPolicy&&(i.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?i.credentials="include":o.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function r(o){if(o.ep)return;o.ep=!0;const i=n(o);fetch(o.href,i)}})();const Tt=`*,:before,:after{box-sizing:border-box;border-width:0;border-style:solid;border-color:#e5e7eb}:before,:after{--tw-content: ""}html{line-height:1.5;-webkit-text-size-adjust:100%;-moz-tab-size:4;-o-tab-size:4;tab-size:4;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,"Apple Color Emoji","Segoe UI Emoji",Segoe UI Symbol,"Noto Color Emoji";font-feature-settings:normal;font-variation-settings:normal}body{margin:0;line-height:inherit}hr{height:0;color:inherit;border-top-width:1px}abbr:where([title]){-webkit-text-decoration:underline dotted;text-decoration:underline dotted}h1,h2,h3,h4,h5,h6{font-size:inherit;font-weight:inherit}a{color:inherit;text-decoration:inherit}b,strong{font-weight:bolder}code,kbd,samp,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace;font-size:1em}small{font-size:80%}sub,sup{font-size:75%;line-height:0;position:relative;vertical-align:baseline}sub{bottom:-.25em}sup{top:-.5em}table{text-indent:0;border-color:inherit;border-collapse:collapse}button,input,optgroup,select,textarea{font-family:inherit;font-size:100%;font-weight:inherit;line-height:inherit;color:inherit;margin:0;padding:0}button,select{text-transform:none}button,[type=button],[type=reset],[type=submit]{-webkit-appearance:button;background-color:transparent;background-image:none}:-moz-focusring{outline:auto}:-moz-ui-invalid{box-shadow:none}progress{vertical-align:baseline}::-webkit-inner-spin-button,::-webkit-outer-spin-button{height:auto}[type=search]{-webkit-appearance:textfield;outline-offset:-2px}::-webkit-search-decoration{-webkit-appearance:none}::-webkit-file-upload-button{-webkit-appearance:button;font:inherit}summary{display:list-item}blockquote,dl,dd,h1,h2,h3,h4,h5,h6,hr,figure,p,pre{margin:0}fieldset{margin:0;padding:0}legend{padding:0}ol,ul,menu{list-style:none;margin:0;padding:0}textarea{resize:vertical}input::-moz-placeholder,textarea::-moz-placeholder{opacity:1;color:#9ca3af}input::placeholder,textarea::placeholder{opacity:1;color:#9ca3af}button,[role=button]{cursor:pointer}:disabled{cursor:default}img,svg,video,canvas,audio,iframe,embed,object{display:block;vertical-align:middle}img,video{max-width:100%;height:auto}[hidden]{display:none}*,:before,:after{--tw-border-spacing-x: 0;--tw-border-spacing-y: 0;--tw-translate-x: 0;--tw-translate-y: 0;--tw-rotate: 0;--tw-skew-x: 0;--tw-skew-y: 0;--tw-scale-x: 1;--tw-scale-y: 1;--tw-pan-x: ;--tw-pan-y: ;--tw-pinch-zoom: ;--tw-scroll-snap-strictness: proximity;--tw-gradient-from-position: ;--tw-gradient-via-position: ;--tw-gradient-to-position: ;--tw-ordinal: ;--tw-slashed-zero: ;--tw-numeric-figure: ;--tw-numeric-spacing: ;--tw-numeric-fraction: ;--tw-ring-inset: ;--tw-ring-offset-width: 0px;--tw-ring-offset-color: #fff;--tw-ring-color: rgb(59 130 246 / .5);--tw-ring-offset-shadow: 0 0 #0000;--tw-ring-shadow: 0 0 #0000;--tw-shadow: 0 0 #0000;--tw-shadow-colored: 0 0 #0000;--tw-blur: ;--tw-brightness: ;--tw-contrast: ;--tw-grayscale: ;--tw-hue-rotate: ;--tw-invert: ;--tw-saturate: ;--tw-sepia: ;--tw-drop-shadow: ;--tw-backdrop-blur: ;--tw-backdrop-brightness: ;--tw-backdrop-contrast: ;--tw-backdrop-grayscale: ;--tw-backdrop-hue-rotate: ;--tw-backdrop-invert: ;--tw-backdrop-opacity: ;--tw-backdrop-saturate: ;--tw-backdrop-sepia: }::backdrop{--tw-border-spacing-x: 0;--tw-border-spacing-y: 0;--tw-translate-x: 0;--tw-translate-y: 0;--tw-rotate: 0;--tw-skew-x: 0;--tw-skew-y: 0;--tw-scale-x: 1;--tw-scale-y: 1;--tw-pan-x: ;--tw-pan-y: ;--tw-pinch-zoom: ;--tw-scroll-snap-strictness: proximity;--tw-gradient-from-position: ;--tw-gradient-via-position: ;--tw-gradient-to-position: ;--tw-ordinal: ;--tw-slashed-zero: ;--tw-numeric-figure: ;--tw-numeric-spacing: ;--tw-numeric-fraction: ;--tw-ring-inset: ;--tw-ring-offset-width: 0px;--tw-ring-offset-color: #fff;--tw-ring-color: rgb(59 130 246 / .5);--tw-ring-offset-shadow: 0 0 #0000;--tw-ring-shadow: 0 0 #0000;--tw-shadow: 0 0 #0000;--tw-shadow-colored: 0 0 #0000;--tw-blur: ;--tw-brightness: ;--tw-contrast: ;--tw-grayscale: ;--tw-hue-rotate: ;--tw-invert: ;--tw-saturate: ;--tw-sepia: ;--tw-drop-shadow: ;--tw-backdrop-blur: ;--tw-backdrop-brightness: ;--tw-backdrop-contrast: ;--tw-backdrop-grayscale: ;--tw-backdrop-hue-rotate: ;--tw-backdrop-invert: ;--tw-backdrop-opacity: ;--tw-backdrop-saturate: ;--tw-backdrop-sepia: }.container{width:100%}@media (min-width: 640px){.container{max-width:640px}}@media (min-width: 768px){.container{max-width:768px}}@media (min-width: 1024px){.container{max-width:1024px}}@media (min-width: 1280px){.container{max-width:1280px}}@media (min-width: 1536px){.container{max-width:1536px}}.pointer-events-none{pointer-events:none}.absolute{position:absolute}.relative{position:relative}.left-0{left:0px}.top-0{top:0px}.z-10{z-index:10}.inline{display:inline}.h-3{height:.75rem}.h-full{height:100%}.w-3{width:.75rem}.w-full{width:100%}.-translate-x-1\\/2{--tw-translate-x: -50%;transform:translate(var(--tw-translate-x),var(--tw-translate-y)) rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))}.-translate-y-1\\/2{--tw-translate-y: -50%;transform:translate(var(--tw-translate-x),var(--tw-translate-y)) rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))}.cursor-move{cursor:move}.cursor-pointer{cursor:pointer}.rounded-full{border-radius:9999px}.rounded-lg{border-radius:.5rem}.border-2{border-width:2px}.border-gray-200{--tw-border-opacity: 1;border-color:rgb(229 231 235 / var(--tw-border-opacity))}.bg-cyan-500{--tw-bg-opacity: 1;background-color:rgb(6 182 212 / var(--tw-bg-opacity))}.bg-slate-50{--tw-bg-opacity: 1;background-color:rgb(248 250 252 / var(--tw-bg-opacity))}.px-2{padding-left:.5rem;padding-right:.5rem}.py-1{padding-top:.25rem;padding-bottom:.25rem}.font-bold{font-weight:700}.text-black{--tw-text-opacity: 1;color:rgb(0 0 0 / var(--tw-text-opacity))}.shadow-sm{--tw-shadow: 0 1px 2px 0 rgb(0 0 0 / .05);--tw-shadow-colored: 0 1px 2px 0 var(--tw-shadow-color);box-shadow:var(--tw-ring-offset-shadow, 0 0 #0000),var(--tw-ring-shadow, 0 0 #0000),var(--tw-shadow)}.filter{filter:var(--tw-blur) var(--tw-brightness) var(--tw-contrast) var(--tw-grayscale) var(--tw-hue-rotate) var(--tw-invert) var(--tw-saturate) var(--tw-sepia) var(--tw-drop-shadow)}:before,:after{box-sizing:border-box;border-width:0;border-style:solid;border-color:#e5e7eb;--tw-content: ""}:host{line-height:1.5;-webkit-text-size-adjust:100%;-moz-tab-size:4;-o-tab-size:4;tab-size:4;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,"Apple Color Emoji","Segoe UI Emoji",Segoe UI Symbol,"Noto Color Emoji";font-feature-settings:normal;margin:0;line-height:inherit}
`;function X(){}function mt(t){return t()}function ut(){return Object.create(null)}function Z(t){t.forEach(mt)}function yt(t){return typeof t=="function"}function Rt(t,e){return t!=t?e==e:t!==e||t&&typeof t=="object"||typeof t=="function"}function Lt(t){return Object.keys(t).length===0}function J(t,e){t.appendChild(e)}function Bt(t,e,n){const r=Ot(t);if(!r.getElementById(e)){const o=F("style");o.id=e,o.textContent=n,It(r,o)}}function Ot(t){if(!t)return document;const e=t.getRootNode?t.getRootNode():t.ownerDocument;return e&&e.host?e:t.ownerDocument}function It(t,e){return J(t.head||t,e),e.sheet}function _t(t,e,n){t.insertBefore(e,n||null)}function lt(t){t.parentNode&&t.parentNode.removeChild(t)}function Dt(t,e){for(let n=0;n<t.length;n+=1)t[n]&&t[n].d(e)}function F(t){return document.createElement(t)}function Ft(t){return document.createTextNode(t)}function Yt(){return Ft(" ")}function z(t,e,n){n==null?t.removeAttribute(e):t.getAttribute(e)!==n&&t.setAttribute(e,n)}function Xt(t){return Array.from(t.childNodes)}function K(t,e,n,r){n==null?t.style.removeProperty(e):t.style.setProperty(e,n,r?"important":"")}let q;function Y(t){q=t}function qt(){if(!q)throw new Error("Function called outside component initialization");return q}function Ut(t){qt().$$.on_mount.push(t)}const R=[],rt=[];let L=[];const ft=[],Vt=Promise.resolve();let ot=!1;function Ht(){ot||(ot=!0,Vt.then(bt))}function it(t){L.push(t)}const et=new Set;let T=0;function bt(){if(T!==0)return;const t=q;do{try{for(;T<R.length;){const e=R[T];T++,Y(e),Kt(e.$$)}}catch(e){throw R.length=0,T=0,e}for(Y(null),R.length=0,T=0;rt.length;)rt.pop()();for(let e=0;e<L.length;e+=1){const n=L[e];et.has(n)||(et.add(n),n())}L.length=0}while(R.length);for(;ft.length;)ft.pop()();ot=!1,et.clear(),Y(t)}function Kt(t){if(t.fragment!==null){t.update(),Z(t.before_update);const e=t.dirty;t.dirty=[-1],t.fragment&&t.fragment.p(t.ctx,e),t.after_update.forEach(it)}}function Gt(t){const e=[],n=[];L.forEach(r=>t.indexOf(r)===-1?e.push(r):n.push(r)),n.forEach(r=>r()),L=e}const Jt=new Set;function Qt(t,e){t&&t.i&&(Jt.delete(t),t.i(e))}function Wt(t,e,n,r){const{fragment:o,after_update:i}=t.$$;o&&o.m(e,n),r||it(()=>{const a=t.$$.on_mount.map(mt).filter(yt);t.$$.on_destroy?t.$$.on_destroy.push(...a):Z(a),t.$$.on_mount=[]}),i.forEach(it)}function Zt(t,e){const n=t.$$;n.fragment!==null&&(Gt(n.after_update),Z(n.on_destroy),n.fragment&&n.fragment.d(e),n.on_destroy=n.fragment=null,n.ctx=[])}function jt(t,e){t.$$.dirty[0]===-1&&(R.push(t),Ht(),t.$$.dirty.fill(0)),t.$$.dirty[e/31|0]|=1<<e%31}function te(t,e,n,r,o,i,a,l=[-1]){const s=q;Y(t);const u=t.$$={fragment:null,ctx:[],props:i,update:X,not_equal:o,bound:ut(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(e.context||(s?s.$$.context:[])),callbacks:ut(),dirty:l,skip_bound:!1,root:e.target||s.$$.root};a&&a(u.root);let f=!1;if(u.ctx=n?n(t,e.props||{},(p,d,..._)=>{const v=_.length?_[0]:d;return u.ctx&&o(u.ctx[p],u.ctx[p]=v)&&(!u.skip_bound&&u.bound[p]&&u.bound[p](v),f&&jt(t,p)),d}):[],u.update(),f=!0,Z(u.before_update),u.fragment=r?r(u.ctx):!1,e.target){if(e.hydrate){const p=Xt(e.target);u.fragment&&u.fragment.l(p),p.forEach(lt)}else u.fragment&&u.fragment.c();e.intro&&Qt(t.$$.fragment),Wt(t,e.target,e.anchor,e.customElement),bt()}Y(s)}class ee{$destroy(){Zt(this,1),this.$destroy=X}$on(e,n){if(!yt(n))return X;const r=this.$$.callbacks[e]||(this.$$.callbacks[e]=[]);return r.push(n),()=>{const o=r.indexOf(n);o!==-1&&r.splice(o,1)}}$set(e){this.$$set&&!Lt(e)&&(this.$$.skip_bound=!0,this.$$set(e),this.$$.skip_bound=!1)}}var ne={value:()=>{}};function vt(){for(var t=0,e=arguments.length,n={},r;t<e;++t){if(!(r=arguments[t]+"")||r in n||/[\s.]/.test(r))throw new Error("illegal type: "+r);n[r]=[]}return new Q(n)}function Q(t){this._=t}function re(t,e){return t.trim().split(/^|\s+/).map(function(n){var r="",o=n.indexOf(".");if(o>=0&&(r=n.slice(o+1),n=n.slice(0,o)),n&&!e.hasOwnProperty(n))throw new Error("unknown type: "+n);return{type:n,name:r}})}Q.prototype=vt.prototype={constructor:Q,on:function(t,e){var n=this._,r=re(t+"",n),o,i=-1,a=r.length;if(arguments.length<2){for(;++i<a;)if((o=(t=r[i]).type)&&(o=oe(n[o],t.name)))return o;return}if(e!=null&&typeof e!="function")throw new Error("invalid callback: "+e);for(;++i<a;)if(o=(t=r[i]).type)n[o]=ht(n[o],t.name,e);else if(e==null)for(o in n)n[o]=ht(n[o],t.name,null);return this},copy:function(){var t={},e=this._;for(var n in e)t[n]=e[n].slice();return new Q(t)},call:function(t,e){if((o=arguments.length-2)>0)for(var n=new Array(o),r=0,o,i;r<o;++r)n[r]=arguments[r+2];if(!this._.hasOwnProperty(t))throw new Error("unknown type: "+t);for(i=this._[t],r=0,o=i.length;r<o;++r)i[r].value.apply(e,n)},apply:function(t,e,n){if(!this._.hasOwnProperty(t))throw new Error("unknown type: "+t);for(var r=this._[t],o=0,i=r.length;o<i;++o)r[o].value.apply(e,n)}};function oe(t,e){for(var n=0,r=t.length,o;n<r;++n)if((o=t[n]).name===e)return o.value}function ht(t,e,n){for(var r=0,o=t.length;r<o;++r)if(t[r].name===e){t[r]=ne,t=t.slice(0,r).concat(t.slice(r+1));break}return n!=null&&t.push({name:e,value:n}),t}var st="http://www.w3.org/1999/xhtml";const dt={svg:"http://www.w3.org/2000/svg",xhtml:st,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"};function xt(t){var e=t+="",n=e.indexOf(":");return n>=0&&(e=t.slice(0,n))!=="xmlns"&&(t=t.slice(n+1)),dt.hasOwnProperty(e)?{space:dt[e],local:t}:t}function ie(t){return function(){var e=this.ownerDocument,n=this.namespaceURI;return n===st&&e.documentElement.namespaceURI===st?e.createElement(t):e.createElementNS(n,t)}}function se(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}function kt(t){var e=xt(t);return(e.local?se:ie)(e)}function ae(){}function Et(t){return t==null?ae:function(){return this.querySelector(t)}}function le(t){typeof t!="function"&&(t=Et(t));for(var e=this._groups,n=e.length,r=new Array(n),o=0;o<n;++o)for(var i=e[o],a=i.length,l=r[o]=new Array(a),s,u,f=0;f<a;++f)(s=i[f])&&(u=t.call(s,s.__data__,f,i))&&("__data__"in s&&(u.__data__=s.__data__),l[f]=u);return new E(r,this._parents)}function ce(t){return t==null?[]:Array.isArray(t)?t:Array.from(t)}function ue(){return[]}function fe(t){return t==null?ue:function(){return this.querySelectorAll(t)}}function he(t){return function(){return ce(t.apply(this,arguments))}}function de(t){typeof t=="function"?t=he(t):t=fe(t);for(var e=this._groups,n=e.length,r=[],o=[],i=0;i<n;++i)for(var a=e[i],l=a.length,s,u=0;u<l;++u)(s=a[u])&&(r.push(t.call(s,s.__data__,u,a)),o.push(s));return new E(r,o)}function pe(t){return function(){return this.matches(t)}}function At(t){return function(e){return e.matches(t)}}var ge=Array.prototype.find;function we(t){return function(){return ge.call(this.children,t)}}function me(){return this.firstElementChild}function ye(t){return this.select(t==null?me:we(typeof t=="function"?t:At(t)))}var _e=Array.prototype.filter;function be(){return Array.from(this.children)}function ve(t){return function(){return _e.call(this.children,t)}}function xe(t){return this.selectAll(t==null?be:ve(typeof t=="function"?t:At(t)))}function ke(t){typeof t!="function"&&(t=pe(t));for(var e=this._groups,n=e.length,r=new Array(n),o=0;o<n;++o)for(var i=e[o],a=i.length,l=r[o]=[],s,u=0;u<a;++u)(s=i[u])&&t.call(s,s.__data__,u,i)&&l.push(s);return new E(r,this._parents)}function St(t){return new Array(t.length)}function Ee(){return new E(this._enter||this._groups.map(St),this._parents)}function W(t,e){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=e}W.prototype={constructor:W,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,e){return this._parent.insertBefore(t,e)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}};function Ae(t){return function(){return t}}function Se(t,e,n,r,o,i){for(var a=0,l,s=e.length,u=i.length;a<u;++a)(l=e[a])?(l.__data__=i[a],r[a]=l):n[a]=new W(t,i[a]);for(;a<s;++a)(l=e[a])&&(o[a]=l)}function Ce(t,e,n,r,o,i,a){var l,s,u=new Map,f=e.length,p=i.length,d=new Array(f),_;for(l=0;l<f;++l)(s=e[l])&&(d[l]=_=a.call(s,s.__data__,l,e)+"",u.has(_)?o[l]=s:u.set(_,s));for(l=0;l<p;++l)_=a.call(t,i[l],l,i)+"",(s=u.get(_))?(r[l]=s,s.__data__=i[l],u.delete(_)):n[l]=new W(t,i[l]);for(l=0;l<f;++l)(s=e[l])&&u.get(d[l])===s&&(o[l]=s)}function Ne(t){return t.__data__}function ze(t,e){if(!arguments.length)return Array.from(this,Ne);var n=e?Ce:Se,r=this._parents,o=this._groups;typeof t!="function"&&(t=Ae(t));for(var i=o.length,a=new Array(i),l=new Array(i),s=new Array(i),u=0;u<i;++u){var f=r[u],p=o[u],d=p.length,_=Me(t.call(f,f&&f.__data__,u,r)),v=_.length,M=l[u]=new Array(v),P=a[u]=new Array(v),O=s[u]=new Array(d);n(f,p,M,P,O,_,e);for(var w=0,m=0,c,h;w<v;++w)if(c=M[w]){for(w>=m&&(m=w+1);!(h=P[m])&&++m<v;);c._next=h||null}}return a=new E(a,r),a._enter=l,a._exit=s,a}function Me(t){return typeof t=="object"&&"length"in t?t:Array.from(t)}function Pe(){return new E(this._exit||this._groups.map(St),this._parents)}function $e(t,e,n){var r=this.enter(),o=this,i=this.exit();return typeof t=="function"?(r=t(r),r&&(r=r.selection())):r=r.append(t+""),e!=null&&(o=e(o),o&&(o=o.selection())),n==null?i.remove():n(i),r&&o?r.merge(o).order():o}function Te(t){for(var e=t.selection?t.selection():t,n=this._groups,r=e._groups,o=n.length,i=r.length,a=Math.min(o,i),l=new Array(o),s=0;s<a;++s)for(var u=n[s],f=r[s],p=u.length,d=l[s]=new Array(p),_,v=0;v<p;++v)(_=u[v]||f[v])&&(d[v]=_);for(;s<o;++s)l[s]=n[s];return new E(l,this._parents)}function Re(){for(var t=this._groups,e=-1,n=t.length;++e<n;)for(var r=t[e],o=r.length-1,i=r[o],a;--o>=0;)(a=r[o])&&(i&&a.compareDocumentPosition(i)^4&&i.parentNode.insertBefore(a,i),i=a);return this}function Le(t){t||(t=Be);function e(p,d){return p&&d?t(p.__data__,d.__data__):!p-!d}for(var n=this._groups,r=n.length,o=new Array(r),i=0;i<r;++i){for(var a=n[i],l=a.length,s=o[i]=new Array(l),u,f=0;f<l;++f)(u=a[f])&&(s[f]=u);s.sort(e)}return new E(o,this._parents).order()}function Be(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}function Oe(){var t=arguments[0];return arguments[0]=this,t.apply(null,arguments),this}function Ie(){return Array.from(this)}function De(){for(var t=this._groups,e=0,n=t.length;e<n;++e)for(var r=t[e],o=0,i=r.length;o<i;++o){var a=r[o];if(a)return a}return null}function Fe(){let t=0;for(const e of this)++t;return t}function Ye(){return!this.node()}function Xe(t){for(var e=this._groups,n=0,r=e.length;n<r;++n)for(var o=e[n],i=0,a=o.length,l;i<a;++i)(l=o[i])&&t.call(l,l.__data__,i,o);return this}function qe(t){return function(){this.removeAttribute(t)}}function Ue(t){return function(){this.removeAttributeNS(t.space,t.local)}}function Ve(t,e){return function(){this.setAttribute(t,e)}}function He(t,e){return function(){this.setAttributeNS(t.space,t.local,e)}}function Ke(t,e){return function(){var n=e.apply(this,arguments);n==null?this.removeAttribute(t):this.setAttribute(t,n)}}function Ge(t,e){return function(){var n=e.apply(this,arguments);n==null?this.removeAttributeNS(t.space,t.local):this.setAttributeNS(t.space,t.local,n)}}function Je(t,e){var n=xt(t);if(arguments.length<2){var r=this.node();return n.local?r.getAttributeNS(n.space,n.local):r.getAttribute(n)}return this.each((e==null?n.local?Ue:qe:typeof e=="function"?n.local?Ge:Ke:n.local?He:Ve)(n,e))}function Ct(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}function Qe(t){return function(){this.style.removeProperty(t)}}function We(t,e,n){return function(){this.style.setProperty(t,e,n)}}function Ze(t,e,n){return function(){var r=e.apply(this,arguments);r==null?this.style.removeProperty(t):this.style.setProperty(t,r,n)}}function je(t,e,n){return arguments.length>1?this.each((e==null?Qe:typeof e=="function"?Ze:We)(t,e,n??"")):tn(this.node(),t)}function tn(t,e){return t.style.getPropertyValue(e)||Ct(t).getComputedStyle(t,null).getPropertyValue(e)}function en(t){return function(){delete this[t]}}function nn(t,e){return function(){this[t]=e}}function rn(t,e){return function(){var n=e.apply(this,arguments);n==null?delete this[t]:this[t]=n}}function on(t,e){return arguments.length>1?this.each((e==null?en:typeof e=="function"?rn:nn)(t,e)):this.node()[t]}function Nt(t){return t.trim().split(/^|\s+/)}function ct(t){return t.classList||new zt(t)}function zt(t){this._node=t,this._names=Nt(t.getAttribute("class")||"")}zt.prototype={add:function(t){var e=this._names.indexOf(t);e<0&&(this._names.push(t),this._node.setAttribute("class",this._names.join(" ")))},remove:function(t){var e=this._names.indexOf(t);e>=0&&(this._names.splice(e,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(t){return this._names.indexOf(t)>=0}};function Mt(t,e){for(var n=ct(t),r=-1,o=e.length;++r<o;)n.add(e[r])}function Pt(t,e){for(var n=ct(t),r=-1,o=e.length;++r<o;)n.remove(e[r])}function sn(t){return function(){Mt(this,t)}}function an(t){return function(){Pt(this,t)}}function ln(t,e){return function(){(e.apply(this,arguments)?Mt:Pt)(this,t)}}function cn(t,e){var n=Nt(t+"");if(arguments.length<2){for(var r=ct(this.node()),o=-1,i=n.length;++o<i;)if(!r.contains(n[o]))return!1;return!0}return this.each((typeof e=="function"?ln:e?sn:an)(n,e))}function un(){this.textContent=""}function fn(t){return function(){this.textContent=t}}function hn(t){return function(){var e=t.apply(this,arguments);this.textContent=e??""}}function dn(t){return arguments.length?this.each(t==null?un:(typeof t=="function"?hn:fn)(t)):this.node().textContent}function pn(){this.innerHTML=""}function gn(t){return function(){this.innerHTML=t}}function wn(t){return function(){var e=t.apply(this,arguments);this.innerHTML=e??""}}function mn(t){return arguments.length?this.each(t==null?pn:(typeof t=="function"?wn:gn)(t)):this.node().innerHTML}function yn(){this.nextSibling&&this.parentNode.appendChild(this)}function _n(){return this.each(yn)}function bn(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}function vn(){return this.each(bn)}function xn(t){var e=typeof t=="function"?t:kt(t);return this.select(function(){return this.appendChild(e.apply(this,arguments))})}function kn(){return null}function En(t,e){var n=typeof t=="function"?t:kt(t),r=e==null?kn:typeof e=="function"?e:Et(e);return this.select(function(){return this.insertBefore(n.apply(this,arguments),r.apply(this,arguments)||null)})}function An(){var t=this.parentNode;t&&t.removeChild(this)}function Sn(){return this.each(An)}function Cn(){var t=this.cloneNode(!1),e=this.parentNode;return e?e.insertBefore(t,this.nextSibling):t}function Nn(){var t=this.cloneNode(!0),e=this.parentNode;return e?e.insertBefore(t,this.nextSibling):t}function zn(t){return this.select(t?Nn:Cn)}function Mn(t){return arguments.length?this.property("__data__",t):this.node().__data__}function Pn(t){return function(e){t.call(this,e,this.__data__)}}function $n(t){return t.trim().split(/^|\s+/).map(function(e){var n="",r=e.indexOf(".");return r>=0&&(n=e.slice(r+1),e=e.slice(0,r)),{type:e,name:n}})}function Tn(t){return function(){var e=this.__on;if(e){for(var n=0,r=-1,o=e.length,i;n<o;++n)i=e[n],(!t.type||i.type===t.type)&&i.name===t.name?this.removeEventListener(i.type,i.listener,i.options):e[++r]=i;++r?e.length=r:delete this.__on}}}function Rn(t,e,n){return function(){var r=this.__on,o,i=Pn(e);if(r){for(var a=0,l=r.length;a<l;++a)if((o=r[a]).type===t.type&&o.name===t.name){this.removeEventListener(o.type,o.listener,o.options),this.addEventListener(o.type,o.listener=i,o.options=n),o.value=e;return}}this.addEventListener(t.type,i,n),o={type:t.type,name:t.name,value:e,listener:i,options:n},r?r.push(o):this.__on=[o]}}function Ln(t,e,n){var r=$n(t+""),o,i=r.length,a;if(arguments.length<2){var l=this.node().__on;if(l){for(var s=0,u=l.length,f;s<u;++s)for(o=0,f=l[s];o<i;++o)if((a=r[o]).type===f.type&&a.name===f.name)return f.value}return}for(l=e?Rn:Tn,o=0;o<i;++o)this.each(l(r[o],e,n));return this}function $t(t,e,n){var r=Ct(t),o=r.CustomEvent;typeof o=="function"?o=new o(e,n):(o=r.document.createEvent("Event"),n?(o.initEvent(e,n.bubbles,n.cancelable),o.detail=n.detail):o.initEvent(e,!1,!1)),t.dispatchEvent(o)}function Bn(t,e){return function(){return $t(this,t,e)}}function On(t,e){return function(){return $t(this,t,e.apply(this,arguments))}}function In(t,e){return this.each((typeof e=="function"?On:Bn)(t,e))}function*Dn(){for(var t=this._groups,e=0,n=t.length;e<n;++e)for(var r=t[e],o=0,i=r.length,a;o<i;++o)(a=r[o])&&(yield a)}var Fn=[null];function E(t,e){this._groups=t,this._parents=e}function Yn(){return this}E.prototype={constructor:E,select:le,selectAll:de,selectChild:ye,selectChildren:xe,filter:ke,data:ze,enter:Ee,exit:Pe,join:$e,merge:Te,selection:Yn,order:Re,sort:Le,call:Oe,nodes:Ie,node:De,size:Fe,empty:Ye,each:Xe,attr:Je,style:je,property:on,classed:cn,text:dn,html:mn,raise:_n,lower:vn,append:xn,insert:En,remove:Sn,clone:zn,datum:Mn,on:Ln,dispatch:In,[Symbol.iterator]:Dn};function U(t){return typeof t=="string"?new E([[document.querySelector(t)]],[document.documentElement]):new E([[t]],Fn)}function Xn(t){let e;for(;e=t.sourceEvent;)t=e;return t}function pt(t,e){if(t=Xn(t),e===void 0&&(e=t.currentTarget),e){var n=e.ownerSVGElement||e;if(n.createSVGPoint){var r=n.createSVGPoint();return r.x=t.clientX,r.y=t.clientY,r=r.matrixTransform(e.getScreenCTM().inverse()),[r.x,r.y]}if(e.getBoundingClientRect){var o=e.getBoundingClientRect();return[t.clientX-o.left-e.clientLeft,t.clientY-o.top-e.clientTop]}}return[t.pageX,t.pageY]}const qn={passive:!1},V={capture:!0,passive:!1};function nt(t){t.stopImmediatePropagation()}function B(t){t.preventDefault(),t.stopImmediatePropagation()}function Un(t){var e=t.document.documentElement,n=U(t).on("dragstart.drag",B,V);"onselectstart"in e?n.on("selectstart.drag",B,V):(e.__noselect=e.style.MozUserSelect,e.style.MozUserSelect="none")}function Vn(t,e){var n=t.document.documentElement,r=U(t).on("dragstart.drag",null);e&&(r.on("click.drag",B,V),setTimeout(function(){r.on("click.drag",null)},0)),"onselectstart"in n?r.on("selectstart.drag",null):(n.style.MozUserSelect=n.__noselect,delete n.__noselect)}const G=t=>()=>t;function at(t,{sourceEvent:e,subject:n,target:r,identifier:o,active:i,x:a,y:l,dx:s,dy:u,dispatch:f}){Object.defineProperties(this,{type:{value:t,enumerable:!0,configurable:!0},sourceEvent:{value:e,enumerable:!0,configurable:!0},subject:{value:n,enumerable:!0,configurable:!0},target:{value:r,enumerable:!0,configurable:!0},identifier:{value:o,enumerable:!0,configurable:!0},active:{value:i,enumerable:!0,configurable:!0},x:{value:a,enumerable:!0,configurable:!0},y:{value:l,enumerable:!0,configurable:!0},dx:{value:s,enumerable:!0,configurable:!0},dy:{value:u,enumerable:!0,configurable:!0},_:{value:f}})}at.prototype.on=function(){var t=this._.on.apply(this._,arguments);return t===this._?this:t};function Hn(t){return!t.ctrlKey&&!t.button}function Kn(){return this.parentNode}function Gn(t,e){return e??{x:t.x,y:t.y}}function Jn(){return navigator.maxTouchPoints||"ontouchstart"in this}function Qn(){var t=Hn,e=Kn,n=Gn,r=Jn,o={},i=vt("start","drag","end"),a=0,l,s,u,f,p=0;function d(c){c.on("mousedown.drag",_).filter(r).on("touchstart.drag",P).on("touchmove.drag",O,qn).on("touchend.drag touchcancel.drag",w).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}function _(c,h){if(!(f||!t.call(this,c,h))){var y=m(this,e.call(this,c,h),c,h,"mouse");y&&(U(c.view).on("mousemove.drag",v,V).on("mouseup.drag",M,V),Un(c.view),nt(c),u=!1,l=c.clientX,s=c.clientY,y("start",c))}}function v(c){if(B(c),!u){var h=c.clientX-l,y=c.clientY-s;u=h*h+y*y>p}o.mouse("drag",c)}function M(c){U(c.view).on("mousemove.drag mouseup.drag",null),Vn(c.view,u),B(c),o.mouse("end",c)}function P(c,h){if(t.call(this,c,h)){var y=c.changedTouches,g=e.call(this,c,h),b=y.length,k,A;for(k=0;k<b;++k)(A=m(this,g,c,h,y[k].identifier,y[k]))&&(nt(c),A("start",c,y[k]))}}function O(c){var h=c.changedTouches,y=h.length,g,b;for(g=0;g<y;++g)(b=o[h[g].identifier])&&(B(c),b("drag",c,h[g]))}function w(c){var h=c.changedTouches,y=h.length,g,b;for(f&&clearTimeout(f),f=setTimeout(function(){f=null},500),g=0;g<y;++g)(b=o[h[g].identifier])&&(nt(c),b("end",c,h[g]))}function m(c,h,y,g,b,k){var A=i.copy(),x=pt(k||y,h),S,I,N;if((N=n.call(c,new at("beforestart",{sourceEvent:y,target:d,identifier:b,active:a,x:x[0],y:x[1],dx:0,dy:0,dispatch:A}),g))!=null)return S=N.x-x[0]||0,I=N.y-x[1]||0,function $(C,H,j){var D=x,tt;switch(C){case"start":o[b]=$,tt=a++;break;case"end":delete o[b],--a;case"drag":x=pt(j||H,h),tt=a;break}A.call(C,c,new at(C,{sourceEvent:H,subject:N,target:d,identifier:b,active:tt,x:x[0]+S,y:x[1]+I,dx:x[0]-D[0],dy:x[1]-D[1],dispatch:A}),g)}}return d.filter=function(c){return arguments.length?(t=typeof c=="function"?c:G(!!c),d):t},d.container=function(c){return arguments.length?(e=typeof c=="function"?c:G(c),d):e},d.subject=function(c){return arguments.length?(n=typeof c=="function"?c:G(c),d):n},d.touchable=function(c){return arguments.length?(r=typeof c=="function"?c:G(!!c),d):r},d.on=function(){var c=i.on.apply(i,arguments);return c===i?d:c},d.clickDistance=function(c){return arguments.length?(p=(c=+c)*c,d):Math.sqrt(p)},d}function Wn(t){Bt(t,"svelte-39b13y","canvas.svelte-39b13y{border-radius:0.5rem;--tw-shadow:0 1px 2px 0 rgb(0 0 0 / 0.05);--tw-shadow-colored:0 1px 2px 0 var(--tw-shadow-color);box-shadow:var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow)}.container.svelte-39b13y *{color:black !important}")}function gt(t,e,n){const r=t.slice();return r[15]=e[n],r}function wt(t){let e;return{c(){e=F("div"),z(e,"class","absolute w-3 h-3 rounded-full bg-cyan-500 -translate-x-1/2 -translate-y-1/2"),K(e,"top",t[15][1]*100+"%"),K(e,"left",t[15][0]*100+"%")},m(n,r){_t(n,e,r)},p(n,r){r&4&&K(e,"top",n[15][1]*100+"%"),r&4&&K(e,"left",n[15][0]*100+"%")},d(n){n&&lt(e)}}}function Zn(t){let e,n,r,o,i,a=t[2],l=[];for(let s=0;s<a.length;s+=1)l[s]=wt(gt(t,a,s));return{c(){e=F("div"),n=F("canvas"),o=Yt(),i=F("div");for(let s=0;s<l.length;s+=1)l[s].c();z(n,"class",r="w-full border-2 border-gray-200 "+(t[1]?"cursor-move":"cursor-pointer")+" svelte-39b13y"),z(n,"width","256"),z(n,"height","256"),z(i,"class","absolute top-0 left-0 w-full h-full pointer-events-none touch-events-none"),z(e,"class","relative container svelte-39b13y")},m(s,u){_t(s,e,u),J(e,n),t[3](n),J(e,o),J(e,i);for(let f=0;f<l.length;f+=1)l[f]&&l[f].m(i,null)},p(s,[u]){if(u&2&&r!==(r="w-full border-2 border-gray-200 "+(s[1]?"cursor-move":"cursor-pointer")+" svelte-39b13y")&&z(n,"class",r),u&4){a=s[2];let f;for(f=0;f<a.length;f+=1){const p=gt(s,a,f);l[f]?l[f].p(p,u):(l[f]=wt(p),l[f].c(),l[f].m(i,null))}for(;f<l.length;f+=1)l[f].d(1);l.length=a.length}},i:X,o:X,d(s){s&&lt(e),t[3](null),Dt(l,s)}}}function jn(t,e){e.value=t;const n=new CustomEvent("input");e.dispatchEvent(n)}function tr(t,e,n){let r,o,i,a,l=!1,s=[],u=[];Ut(()=>{i=document.getElementById("canvas-root"),i._data||(i._data={image:null}),i.dataset.mode&&i.dataset.mode,o=r.getContext("2d"),a=document.querySelector("#dxdysxsy textarea"),U(r).on("dblclick",d).on("touchstart",p).call(P()),i.loadBase64Image=M,i.resetStopPoints=v});let f=0;function p(w){let m=Date.now();if(m-f<500){const c=r.getBoundingClientRect(),h=r.width/c.width,y=r.height/c.height,g=(w.touches[0].clientX-c.left)*h,b=(w.touches[0].clientY-c.top)*y;_(g,b)}f=m}function d(w){const m=r.getBoundingClientRect(),c=r.width/m.width,h=r.height/m.height,y=w.offsetX*c,g=w.offsetY*h;_(y,g)}function _(w,m){const c=s.filter(h=>Math.sqrt((h[0]-w)**2+(h[1]-m)**2)>10);c.length<s.length?s=c:s=s.concat([[w,m]]),n(2,u=s.map(h=>[h[0]/r.width,h[1]/r.height]))}function v(){s=[],n(2,u=[])}async function M(w){const m=new Image;m.src=w,m.onload=()=>{o.drawImage(m,0,0,r.width,r.height)}}function P(){let w=0,m=0;function c(g){const b=r.getBoundingClientRect(),k=r.width/b.width,A=r.height/b.height,x=g.x*k,S=g.y*A;w=x,m=S,n(1,l=!0)}function h(g){const b=r.getBoundingClientRect(),k=r.width/b.width,A=r.height/b.height,x=g.x*k,S=g.y*A,I=g.subject.x*k,N=g.subject.y*A;let $=Math.floor(x-I),C=Math.floor(S-N);const H=Math.floor(I),j=Math.floor(N);$=Math.sign($)*Math.min(Math.abs($),255),C=Math.sign(C)*Math.min(Math.abs(C),255);const D=JSON.stringify({dx:$,dy:C,sx:H,sy:j,stopPoints:s});Math.sqrt((w-x)**2+(m-S)**2)>5&&(jn(D,a),console.log("dragged",D),w=x,m=S)}function y(g){n(1,l=!1)}return Qn().on("start",c).on("drag",h).on("end",y)}function O(w){rt[w?"unshift":"push"](()=>{r=w,n(0,r)})}return[r,l,u,O]}class er extends ee{constructor(e){super(),te(this,e,tr,Zn,Rt,{},Wn)}}class nr extends HTMLElement{constructor(){super()}connectedCallback(){const e=this.attachShadow({mode:"open"}),n=document.createElement("style");n.appendChild(document.createTextNode(Tt)),e.appendChild(n),new er({target:e})}}customElements.define("draggan-canvas",nr);
