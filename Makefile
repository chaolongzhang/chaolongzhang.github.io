### Build and deploy http://xx

SSH_USER = zhang@xx
DOCUMENT_ROOT = ~/xxx
PUBLIC_DIR = public/

all: deploy

staging: site
	rsync -crzve 'ssh -p 22' $(PUBLIC_DIR) $(STAGING_USER):$(STAGING_ROOT)

server: css
	hugo server -ws .

deploy: site
	rsync --exclude='.DS_Store' -Prvzce 'ssh -p 22' $(PUBLIC_DIR) $(SSH_USER):$(DOCUMENT_ROOT) 

depdel: site
	rsync --exclude='.DS_Store' -Prvzce 'ssh -p 22' --delete-after $(PUBLIC_DIR) $(SSH_USER):$(DOCUMENT_ROOT) 


site: .FORCE
	hugo 
	find public -type d -print0 | xargs -0 chmod 755
	find public -type f -print0 | xargs -0 chmod 644

clean:
	rm -rf public/

.FORCE:
