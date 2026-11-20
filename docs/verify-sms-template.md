# Verify SMS template: recommended wording

This is the recommended wording for the one-time code text, which carries a fraud warning. The blueprint
turns on Verify's `do_not_share_warning_enabled` on its Verify Service, which needs no
template. The full wording below needs a **Twilio-approved custom template** that you request
from Twilio (Twilio must enable custom templates on your account). Once approved, pass its SID to
`scripts/twilio_setup.py --verify-template-sid HJ...`, which sets it as the service's
`DefaultTemplateSid`. Until then the service sends Verify's default text plus the
do-not-share warning.

What the wording must do: a short line stating the message is sent because of a login attempt at
OwlBank, and a warning never to share the code, even with the bank's own staff, because anyone
asking for it is attempting fraud.

## Variables

Verify fills these in (Twilio's template docs): `{{friendly_name}}` (the Verify Service's
friendly name, "OwlBank" by default), `{{code}}` (the 6-digit code) and `{{ttl}}` (minutes until
it expires, 10 by default). The blueprint sends no custom substitutions.

## English (locale `en`)

```
{{friendly_name}}: your code is {{code}}. Sent for a login attempt. Never share it with anyone, even bank staff: anyone asking is attempting fraud. Expires in {{ttl}} min.
```

154 characters once filled in (with "OwlBank", a 6-digit code and `10`). All characters are in the
GSM-7 set, so it is **1 segment** (limit 160), with room for a friendly name of up to about 13 characters.

## Spanish (locale `es`)

Recommended, written without accented letters outside GSM-7 (it uses "clave", not "código"):

```
{{friendly_name}}: su clave es {{code}}. Intento de acceso. Nunca la comparta con nadie, ni con el banco: quien la pida intenta un fraude. Vence en {{ttl}} min.
```

143 characters filled in, GSM-7, **1 segment**. It has room for a friendly name of up to about 24 characters; a longer one can push it to 2 segments (153 characters each).

Alternative with the usual spelling:

```
{{friendly_name}}: su código es {{code}}. Intento de acceso. Nunca lo comparta con nadie, ni con el banco: quien lo pida intenta un fraude. Vence en {{ttl}} min.
```

144 characters filled in. The "ó" is not in GSM-7, so the whole message is sent as UCS-2, which
allows 70 characters in one segment and 67 per segment when split: this version is **3 segments**
(about three times the cost of the other). Any of á, í, ó, ú has the same effect. Use it only if
the accented spelling matters more than the cost.

## Segment counts

| Version | Characters (filled in) | Encoding | Segments |
| --- | --- | --- | --- |
| English | 154 | GSM-7 | 1 |
| Spanish, no accents | 143 | GSM-7 | 1 |
| Spanish, accented | 144 | UCS-2 | 3 |

Counts assume friendly name "OwlBank", a 6-digit code and `{{ttl}}` = `10`. GSM-7 limits: 160
characters in one segment, 153 per segment when split. Characters from the GSM-7 extension table
(for example `|`, `{`, `[`, `~`, `^`, `\`, `€`) count as two. A smart apostrophe or quote also forces
UCS-2, so keep straight ASCII punctuation if you edit the text.

## When you request it

- Twilio requires approval of custom templates (see
  [its support article](https://support.twilio.com/hc/en-us/articles/9960174409627-How-to-Customize-the-Verification-Message-for-Twilio-Verify)
  for the timing).
- Ask for both locales (`en`, `es`) in one template, with the same variables.
- The compliance team of the deploying customer should confirm the wording and the consent
  basis for texting (see "One-time codes" in `docs/production-guide.md`).
- The text must not name a phone number or any customer data: Verify fills in only the
  variables above.
